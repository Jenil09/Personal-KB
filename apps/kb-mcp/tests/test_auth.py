"""`StaticTokenVerifier` and `authorize` — the two halves of the auth story.

The verifier is what the SDK's `RequireAuthMiddleware` calls before any tool is
chosen; `authorize` is what a tool body calls afterwards. They are tested apart
here and together in `test_transport.py`, which drives both through a real ASGI
request.

The registry itself — `hmac.compare_digest`, `key_id` attribution, the
`scopes=None` meaning — belongs to `platform_core.auth` and is tested there. What
these assert is the mapping onto MCP's own vocabulary, and the one place it could
invert: an unrestricted key has `scopes=None`, `AccessToken.scopes` cannot spell
that, and reading `[]` back as "no permissions" would turn the permissive default
into the restrictive one.
"""

from contextvars import Token

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from pydantic import SecretStr

from kb_mcp.auth import StaticTokenVerifier, authorize
from platform_core import ApiKeyRegistry, AuthorizationError


@pytest.fixture
def registry() -> ApiKeyRegistry:
    return ApiKeyRegistry(
        {
            "claude": SecretStr("host-token"),
            "reader": SecretStr("read-only-token"),
            "open": SecretStr("unscoped-token"),
        },
        {"claude": frozenset({"search", "write"}), "reader": frozenset({"search"})},
    )


@pytest.fixture
def verifier(registry: ApiKeyRegistry) -> StaticTokenVerifier:
    return StaticTokenVerifier(registry)


def _as_caller(key_id: str, scopes: list[str]) -> Token[AuthenticatedUser | None]:
    """Put a verified token in the context the way `AuthContextMiddleware` does."""
    return auth_context_var.set(
        AuthenticatedUser(AccessToken(token="opaque", client_id=key_id, scopes=scopes))
    )


async def test_a_known_token_resolves_to_its_key_id(verifier: StaticTokenVerifier) -> None:
    token = await verifier.verify_token("host-token")

    assert token is not None
    # `client_id` *is* AD-011's key_id — the same string the audit trail uses.
    assert token.client_id == "claude"
    assert set(token.scopes) == {"search", "write"}


async def test_an_unknown_token_is_none_rather_than_an_exception(
    verifier: StaticTokenVerifier,
) -> None:
    """A verifier reports; it does not raise. The SDK turns `None` into a 401."""
    assert await verifier.verify_token("not-a-key") is None


async def test_an_empty_token_is_rejected(verifier: StaticTokenVerifier) -> None:
    assert await verifier.verify_token("") is None


async def test_a_key_with_no_scope_entry_arrives_unrestricted(
    verifier: StaticTokenVerifier, registry: ApiKeyRegistry
) -> None:
    """`scopes=None` cannot survive as `AccessToken.scopes`, so `authorize` re-resolves.

    The token carries an empty list — there is nothing else it could carry — and
    the check still has to allow everything, because an unlisted key gets every
    scope (AD-024's permissive default, and `kb-api`'s).
    """
    token = await verifier.verify_token("unscoped-token")
    assert token is not None
    assert token.scopes == []

    reset = _as_caller("open", [])
    try:
        authorize(registry, "write")
    finally:
        auth_context_var.reset(reset)


async def test_a_read_only_key_may_not_write(registry: ApiKeyRegistry) -> None:
    reset = _as_caller("reader", ["search"])
    try:
        authorize(registry, "search")
        with pytest.raises(AuthorizationError) as caught:
            authorize(registry, "write")
    finally:
        auth_context_var.reset(reset)

    assert "write" in str(caught.value)
    assert caught.value.context["key_id"] == "reader"


async def test_no_auth_context_allows(registry: ApiKeyRegistry) -> None:
    """Every in-memory test, and nothing that is deployed.

    `build_app` always passes both `auth` and `token_verifier`, so a running
    server has already refused an anonymous caller with a 401 before a tool is
    reached. Allowing here is what keeps tool bodies drivable without a
    transport.
    """
    assert auth_context_var.get() is None
    authorize(registry, "write")


async def test_a_token_whose_key_the_registry_lost_fails_closed(registry: ApiKeyRegistry) -> None:
    """An unresolvable `client_id` is denied, not treated as unrestricted.

    Unreachable through the verifier — the key was matched moments ago in the
    same request — which is precisely why it fails closed: an unreachable branch
    in an authorisation check is the one nobody notices going the wrong way.
    """
    reset = _as_caller("deleted-key", ["search"])
    try:
        with pytest.raises(AuthorizationError):
            authorize(registry, "search")
    finally:
        auth_context_var.reset(reset)
