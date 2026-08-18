"""The HTTP half of bearer authentication: read the header, resolve, attribute.

`Principal` and `ApiKeyRegistry` live in `platform_core.auth` — they are needed
by services that hold no web framework — and are re-exported here so this
module's callers, `kb-api` among them, see one place for the whole subject.

The `key_id` a registry resolves is put on the ASGI scope, where the tier-1
audit middleware (AD-013) can reach it without a dependency of its own.

`/health` is unauthenticated. That is not a rule enforced here — `create_app`
simply mounts the health router without this dependency and every `/v1` router
with it, so a new router cannot be added unauthenticated by forgetting a flag.
"""

from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from platform_core import AuthenticationError, AuthorizationError
from platform_core.auth import ApiKeyRegistry, Principal

__all__ = [
    "ApiKeyRegistry",
    "CurrentPrincipal",
    "Principal",
    "require_api_key",
    "require_scope",
]

PRINCIPAL_STATE_ATTR = "principal"

_bearer = HTTPBearer(
    scheme_name="API key",
    description="`Authorization: Bearer <api key>`. Required on every route except `/health`.",
    auto_error=False,
)


async def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None:
        # `credentials is None` covers a wrong scheme and an empty token alike,
        # so the message describes the shape rather than guessing which it was.
        raise AuthenticationError(
            "Authorization header is missing."
            if "authorization" not in request.headers
            else "Authorization header must be 'Bearer <api key>'."
        )

    registry: ApiKeyRegistry = request.app.state.api_key_registry
    principal = registry.resolve(credentials.credentials)
    setattr(request.state, PRINCIPAL_STATE_ATTR, principal)
    return principal


CurrentPrincipal = Annotated[Principal, Depends(require_api_key)]
"""Declare this on a handler to read the caller; the dependency is already
resolved by the router, so this costs a cache lookup rather than a second check."""


def require_scope(scope: str) -> Callable[[Principal], Coroutine[Any, Any, Principal]]:
    """A dependency asserting the caller holds `scope` (AD-024).

    Declared per route rather than per router, because the split that matters
    runs through `/v1/documents`: listing and reading are `search`, ingesting and
    deleting are `write`, and they share a prefix. Mounting the check higher up
    would force the two halves into separate routers to say something the route
    can say itself.

    The failure is a `403` naming the missing scope. That is deliberately more
    informative than the `401` above it — see `AuthorizationError`.
    """

    async def dependency(principal: CurrentPrincipal) -> Principal:
        if not principal.has_scope(scope):
            raise AuthorizationError(
                f"This API key does not carry the {scope!r} scope.",
                context={"required_scope": scope},
            )
        return principal

    return dependency
