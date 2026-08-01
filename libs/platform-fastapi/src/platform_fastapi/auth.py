"""Bearer-token authentication with caller attribution (AD-011) and scopes (AD-024).

Keys come from environment configuration as `key_id:secret` pairs. The secret
authenticates; the `key_id` is what makes the audit trail worth keeping, so it
is put on the ASGI scope where the tier-1 audit middleware (AD-013) can reach
it without a dependency of its own.

Each key also carries a scope set, which is what stops the n8n key — held by a
workflow that processes scraped, untrusted content — from being able to delete
the corpus. Attribution says afterwards which key did it; scopes are what makes
it not happen. The vocabulary is the service's own (`search`, `write` in
`kb-api`), so nothing here enumerates the valid names: this module only compares
what a route requires against what a key was granted.

`/health` is unauthenticated. That is not a rule enforced here — `create_app`
simply mounts the health router without this dependency and every `/v1` router
with it, so a new router cannot be added unauthenticated by forgetting a flag.
"""

import hmac
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from platform_core import AuthenticationError, AuthorizationError

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


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. `key_id` is recorded on every audit row."""

    key_id: str
    scopes: frozenset[str] | None = None
    """What this key may do, or `None` for a key with no scope entry.

    `None` means unrestricted rather than "no scopes". The distinction is the
    whole reason this is not just an empty set: an operator who configures keys
    and forgets to configure scopes gets a working service, not one where every
    valid key is refused. `HttpServiceSettings.api_key_scopes` documents the
    trade-off; `create_app` logs the resolved grants at startup so the
    permissive case is visible rather than assumed.
    """

    def has_scope(self, scope: str) -> bool:
        return self.scopes is None or scope in self.scopes


class ApiKeyRegistry:
    """Resolves a presented key to the caller it belongs to, with its grants."""

    def __init__(
        self,
        api_keys: Mapping[str, SecretStr],
        scopes: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        granted = scopes or {}
        self._keys = tuple(
            (key_id, secret.get_secret_value().encode(), granted.get(key_id))
            for key_id, secret in api_keys.items()
        )

    @property
    def grants(self) -> tuple[tuple[str, frozenset[str] | None], ...]:
        """Every key id and what it was granted, for the startup log."""
        return tuple((key_id, scopes) for key_id, _, scopes in self._keys)

    def identify(self, presented: str) -> Principal | None:
        """Resolve without raising. `None` for a key that is not recognised.

        The rate limiter needs the caller's identity *before* the router's auth
        dependency has run, and it must not turn an unrecognised key into a
        `403`-shaped failure of its own — that request has a `401` coming, and
        the limiter's job is only to decide whose counter it belongs to.
        """
        candidate = presented.encode()
        matched: Principal | None = None
        for key_id, secret, scopes in self._keys:
            if hmac.compare_digest(candidate, secret):
                matched = Principal(key_id=key_id, scopes=scopes)
        return matched

    def resolve(self, presented: str) -> Principal:
        # No early exit inside `identify`: every configured key is compared on
        # every attempt, so how long the check takes says nothing about which
        # key matched.
        matched = self.identify(presented)
        if matched is None:
            raise AuthenticationError("The presented API key is not recognised.")
        return matched


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
