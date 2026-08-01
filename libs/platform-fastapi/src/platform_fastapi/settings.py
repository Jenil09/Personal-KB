"""Settings `create_app` needs from the operator.

A service subclasses this instead of `BaseServiceSettings` and adds its own
fields; the HTTP-layer configuration — keys, CORS, health budget — comes for
free and is spelled the same way in every service.

Both list-shaped fields are `NoDecode` because pydantic-settings would
otherwise require JSON in the environment. These are values a human types into
a `.env` file, so they are comma-separated instead.
"""

from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import NoDecode

from platform_core import BaseServiceSettings

__all__ = ["HttpServiceSettings"]


class HttpServiceSettings(BaseServiceSettings):
    service_name: str = "service"
    service_version: str = "0.0.0"

    # `KB_API__API_KEYS=n8n:s3cr3t,cli:0th3r` — `key_id:secret` pairs. The
    # key_id is what lands on every audit row (AD-011); the secret never
    # leaves this process.
    api_keys: Annotated[dict[str, SecretStr], NoDecode]

    # `KB_API__API_KEY_SCOPES=n8n:search,cli:search|write` — what each key is
    # allowed to do (AD-024). Kept as its own setting rather than a third field
    # on `api_keys`, because a secret may legitimately contain a colon and
    # there is no unambiguous way to parse `key_id:secret:scopes` when it does.
    #
    # **A key not named here gets every scope.** That is the permissive
    # direction, chosen because the alternative — an unlisted key having no
    # scopes — makes an operator's first deploy fail with a 403 on a valid key,
    # which reads as a bug rather than as configuration. `create_app` logs each
    # key and its resolved scopes at startup so an omission is visible in the
    # first ten lines of the log rather than at the first denied request.
    api_key_scopes: Annotated[dict[str, frozenset[str]], NoDecode] = Field(default_factory=dict)

    # `KB_API__CORS_ORIGINS=https://n8n.example.com`. Empty means no CORS
    # middleware at all, which is the right default for a machine-to-machine API.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()

    health_timeout_seconds: float = Field(default=2.0, gt=0)

    # Technical Design §8: 10 MB, 20x the largest expected document. Nginx caps
    # bodies too; this one holds when nothing is in front of the app.
    max_body_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value: Any) -> Any:
        # `Any` because pydantic hands the raw environment string here but the
        # same field is constructed from a dict in tests and in-process wiring.
        if not isinstance(value, str):
            return value
        keys: dict[str, str] = {}
        for pair in _split(value):
            key_id, separator, secret = pair.partition(":")
            if not separator or not key_id.strip() or not secret.strip():
                raise ValueError(f"expects comma-separated key_id:secret pairs, got {pair!r}")
            keys[key_id.strip()] = secret.strip()
        if not keys:
            raise ValueError("must define at least one key_id:secret pair")
        return keys

    @field_validator("api_key_scopes", mode="before")
    @classmethod
    def _parse_api_key_scopes(cls, value: Any) -> Any:
        # `Any` for the same reason as `api_keys`: an environment string here,
        # a mapping when constructed in tests and in-process wiring.
        if not isinstance(value, str):
            return value
        scopes: dict[str, frozenset[str]] = {}
        for pair in _split(value):
            key_id, separator, granted = pair.partition(":")
            if not separator or not key_id.strip():
                raise ValueError(f"expects comma-separated key_id:scope|scope pairs, got {pair!r}")
            names = frozenset(name.strip() for name in granted.split("|") if name.strip())
            if not names:
                # An empty scope list is almost certainly a typo, and reading it
                # as "no permissions" would produce a key that authenticates and
                # can do nothing — the least diagnosable outcome available.
                raise ValueError(f"key {key_id.strip()!r} was given no scopes")
            scopes[key_id.strip()] = names
        return scopes

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        return _split(value) if isinstance(value, str) else value


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
