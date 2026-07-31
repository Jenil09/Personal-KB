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

    # `KB_API__CORS_ORIGINS=https://n8n.example.com`. Empty means no CORS
    # middleware at all, which is the right default for a machine-to-machine API.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()

    health_timeout_seconds: float = Field(default=2.0, gt=0)

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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        return _split(value) if isinstance(value, str) else value


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
