"""Bearer-token authentication with caller attribution (AD-011).

Keys come from environment configuration as `key_id:secret` pairs. The secret
authenticates; the `key_id` is what makes the audit trail worth keeping, so it
is put on the ASGI scope where the tier-1 audit middleware (AD-013) can reach
it without a dependency of its own.

`/health` is unauthenticated. That is not a rule enforced here — `create_app`
simply mounts the health router without this dependency and every `/v1` router
with it, so a new router cannot be added unauthenticated by forgetting a flag.
"""

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from platform_core import AuthenticationError

__all__ = ["ApiKeyRegistry", "CurrentPrincipal", "Principal", "require_api_key"]

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


class ApiKeyRegistry:
    """Resolves a presented key to the caller it belongs to."""

    def __init__(self, api_keys: Mapping[str, SecretStr]) -> None:
        self._keys = tuple(
            (key_id, secret.get_secret_value().encode()) for key_id, secret in api_keys.items()
        )

    def resolve(self, presented: str) -> Principal:
        candidate = presented.encode()
        matched: str | None = None
        for key_id, secret in self._keys:
            # No early exit: every configured key is compared on every attempt,
            # so how long the check takes says nothing about which key matched.
            if hmac.compare_digest(candidate, secret):
                matched = key_id
        if matched is None:
            raise AuthenticationError("The presented API key is not recognised.")
        return Principal(key_id=matched)


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
