"""Bearer-token authentication with caller attribution (AD-011) and scopes (AD-024).

Keys come from environment configuration as `key_id:secret` pairs. The secret
authenticates; the `key_id` is what makes the audit trail worth keeping, so it
travels with the caller rather than being looked up again later.

Each key also carries a scope set, which is what stops the n8n key — held by a
workflow that processes scraped, untrusted content — from being able to delete
the corpus. Attribution says afterwards which key did it; scopes are what makes
it not happen. The vocabulary is the service's own (`search`, `write` in
`kb-api`), so nothing here enumerates the valid names: this module only compares
what a caller requires against what a key was granted.

The configuration format lives here too, as `ApiKeys` and `ApiKeyScopes`. A
service declares the annotated field and inherits the parsing; two services
spelling `KB_API__API_KEYS` and `KB_MCP__API_KEYS` two ways is how one operator's
mental model stops matching the other service.

Nothing here is HTTP. `platform-fastapi` builds the dependency that reads the
`Authorization` header on top of it; `kb-mcp` builds an MCP `TokenVerifier` on
the same primitives without carrying a web framework or a database driver.
"""

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BeforeValidator, SecretStr
from pydantic_settings import NoDecode

from platform_core.errors import AuthenticationError

__all__ = [
    "ApiKeyRegistry",
    "ApiKeyScopes",
    "ApiKeys",
    "Principal",
    "parse_api_key_scopes",
    "parse_api_keys",
]


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. `key_id` is recorded on every audit row."""

    key_id: str
    scopes: frozenset[str] | None = None
    """What this key may do, or `None` for a key with no scope entry.

    `None` means unrestricted rather than "no scopes". The distinction is the
    whole reason this is not just an empty set: an operator who configures keys
    and forgets to configure scopes gets a working service, not one where every
    valid key is refused. `ApiKeyScopes` documents the trade-off; a service's
    composition root logs the resolved grants at startup so the permissive case
    is visible rather than assumed.
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


def parse_api_keys(value: Any) -> Any:
    """`n8n:s3cr3t,cli:0th3r` — `key_id:secret` pairs, as a human types them.

    `Any` because pydantic hands the raw environment string here but the same
    field is constructed from a mapping in tests and in-process wiring.
    """
    if not isinstance(value, str):
        return value
    keys: dict[str, str] = {}
    for pair in _split(value):
        # One `partition`, so a secret may contain colons of its own.
        key_id, separator, secret = pair.partition(":")
        if not separator or not key_id.strip() or not secret.strip():
            raise ValueError(f"expects comma-separated key_id:secret pairs, got {pair!r}")
        keys[key_id.strip()] = secret.strip()
    if not keys:
        raise ValueError("must define at least one key_id:secret pair")
    return keys


def parse_api_key_scopes(value: Any) -> Any:
    """`n8n:search,cli:search|write` — what each key is allowed to do (AD-024).

    Kept as its own setting rather than a third field on the key pairs, because
    a secret may legitimately contain a colon and there is no unambiguous way to
    parse `key_id:secret:scopes` when it does.
    """
    if not isinstance(value, str):
        return value
    scopes: dict[str, frozenset[str]] = {}
    for pair in _split(value):
        key_id, separator, granted = pair.partition(":")
        if not separator or not key_id.strip():
            raise ValueError(f"expects comma-separated key_id:scope|scope pairs, got {pair!r}")
        names = frozenset(name.strip() for name in granted.split("|") if name.strip())
        if not names:
            # An empty scope list is almost certainly a typo, and reading it as
            # "no permissions" would produce a key that authenticates and can do
            # nothing — the least diagnosable outcome available.
            raise ValueError(f"key {key_id.strip()!r} was given no scopes")
        scopes[key_id.strip()] = names
    return scopes


# Both are `NoDecode` because pydantic-settings would otherwise require JSON in
# the environment. These are values a human types into a `.env` file, so they are
# comma-separated instead. Declared as annotated types rather than as validators
# a service restates: the format is the shared thing, not just the parser.
ApiKeys = Annotated[dict[str, SecretStr], NoDecode, BeforeValidator(parse_api_keys)]
ApiKeyScopes = Annotated[dict[str, frozenset[str]], NoDecode, BeforeValidator(parse_api_key_scopes)]


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
