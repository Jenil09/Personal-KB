"""Settings `create_app` needs from the operator.

A service subclasses this instead of `BaseServiceSettings` and adds its own
fields; the HTTP-layer configuration — keys, CORS, health budget — comes for
free and is spelled the same way in every service.

The key fields are `platform_core.auth`'s annotated types rather than fields
this class parses itself: `kb-mcp` configures inbound keys the same way, and one
format described in two places is one that eventually differs. `cors_origins` is
`NoDecode` for the same reason they are — a human types a comma-separated list
into a `.env` file, not JSON.
"""

from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from platform_core import ApiKeys, ApiKeyScopes, BaseServiceSettings

__all__ = ["HttpServiceSettings"]


class HttpServiceSettings(BaseServiceSettings):
    service_name: str = "service"
    service_version: str = "0.0.0"

    # `KB_API__API_KEYS=n8n:s3cr3t,cli:0th3r` — `key_id:secret` pairs. The
    # key_id is what lands on every audit row (AD-011); the secret never
    # leaves this process.
    api_keys: ApiKeys

    # `KB_API__API_KEY_SCOPES=n8n:search,cli:search|write` — what each key is
    # allowed to do (AD-024).
    #
    # **A key not named here gets every scope.** That is the permissive
    # direction, chosen because the alternative — an unlisted key having no
    # scopes — makes an operator's first deploy fail with a 403 on a valid key,
    # which reads as a bug rather than as configuration. `create_app` logs each
    # key and its resolved scopes at startup so an omission is visible in the
    # first ten lines of the log rather than at the first denied request.
    api_key_scopes: ApiKeyScopes = Field(default_factory=dict)

    # `KB_API__CORS_ORIGINS=https://n8n.example.com`. Empty means no CORS
    # middleware at all, which is the right default for a machine-to-machine API.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()

    health_timeout_seconds: float = Field(default=2.0, gt=0)

    # Technical Design §8: 10 MB, 20x the largest expected document. Nginx caps
    # bodies too; this one holds when nothing is in front of the app.
    max_body_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        return _split(value) if isinstance(value, str) else value


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
