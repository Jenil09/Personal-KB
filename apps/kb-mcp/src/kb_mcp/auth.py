"""Bearer tokens, on `platform_core`'s registry rather than a second implementation.

Two checks, at two levels, and they are not redundant:

**The transport check** is the SDK's `RequireAuthMiddleware`, which calls
`StaticTokenVerifier.verify_token` and rejects an unrecognised token with a `401`
before any tool is chosen. Its `required_scopes` is deliberately empty — it is
enforced endpoint-wide, so naming `search` there would give a write-only key a
`403` on the whole transport and could never express a per-tool split.

**The tool check** is `authorize`, called in each tool body: reads want `search`,
ingest wants `write`, mirroring AD-024. Because the floor above is empty, this is
the only thing enforcing scopes at all — load-bearing, not belt-and-braces.

`ApiKeyRegistry` and `Principal` come from `platform_core`, never
`platform_fastapi`: that package depends on `platform-db`, and importing it would
put SQLAlchemy and asyncpg in an image with no database.

The round trip through `client_id` is the reason `authorize` re-resolves rather
than reading `AccessToken.scopes`. `Principal.scopes is None` means *unrestricted*
(AD-024's permissive default), and `AccessToken.scopes` is a plain `list[str]`
with no way to spell that — an unrestricted key would arrive as `[]` and read as
"no permissions", which is the exact inversion the `None` exists to prevent.
"""

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from platform_core import ApiKeyRegistry, AuthorizationError, Principal

__all__ = ["StaticTokenVerifier", "authorize"]


class StaticTokenVerifier:
    """The SDK's `TokenVerifier` protocol over configured keys.

    Structural, not a subclass: `TokenVerifier` is a `Protocol`, and the SDK
    ships nothing implementing it — `StaticTokenVerifier` was an example class in
    the 1.x repository, not a public API.
    """

    def __init__(self, registry: ApiKeyRegistry) -> None:
        self._registry = registry

    async def verify_token(self, token: str) -> AccessToken | None:
        """`None` for a token that is not recognised — a verifier reports, it does not raise.

        `identify` rather than `resolve` for that reason, and it is also the one
        that compares every configured key with `hmac.compare_digest` and no early
        exit, so how long this takes says nothing about which key matched.
        """
        principal = self._registry.identify(token)
        if principal is None:
            return None
        return AccessToken(
            token=token,
            # AD-011's `key_id` and AD-024's scopes, mapped onto the protocol's
            # own names. Deliberate: `client_id` is what a server log attributes
            # a call to, and it should be the same string the audit trail uses.
            client_id=principal.key_id,
            scopes=sorted(principal.scopes) if principal.scopes is not None else [],
        )


def authorize(registry: ApiKeyRegistry, scope: str) -> None:
    """Raise unless the calling key holds `scope`.

    No authenticated caller means no auth context — a server built without
    `auth` and `token_verifier`, which is every in-memory test and nothing that
    is deployed, because `build_app` always passes both. Allowing it keeps the
    tool bodies testable without a transport; it cannot widen a running server,
    where `RequireAuthMiddleware` has already refused anonymous callers with a
    `401` before a tool is reached.
    """
    token = get_access_token()
    if token is None:
        return
    principal = _principal_for(registry, token.client_id)
    # A `client_id` the registry no longer knows is denied rather than allowed.
    # It should be unreachable — the verifier matched that key moments ago, in
    # this same request — and an unreachable branch in an authorisation check is
    # exactly the one that should fail closed.
    if principal is None or not principal.has_scope(scope):
        raise AuthorizationError(
            f"This key is not allowed to {scope}.",
            context={"key_id": token.client_id, "required_scope": scope},
        )


def _principal_for(registry: ApiKeyRegistry, key_id: str) -> Principal | None:
    """The grants behind a `client_id`, or `None` if the key is gone.

    `grants` rather than the secret, because the token has already been verified
    and re-matching it would be a second comparison of a credential this function
    has no reason to touch.
    """
    for candidate, scopes in registry.grants:
        if candidate == key_id:
            return Principal(key_id=candidate, scopes=scopes)
    return None
