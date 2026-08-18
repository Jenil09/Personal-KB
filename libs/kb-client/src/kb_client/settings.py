"""Where the service is and which key reaches it — nothing else.

`KbClient` reads exactly these five fields. They live here rather than in a
consumer's settings model so that `kb-cli` and `kb-mcp` describe the same
connection the same way, and so the client can be typed against what it actually
uses instead of against whichever consumer happens to construct it.

**Subclassed or nested, not read on its own.** `KbCliSettings` extends this and
overrides the prefix with `KB_CLI__`; `kb-mcp` nests it under its own settings.
The `KB_CLIENT__` prefix here is the fallback for the case nobody intends: an
unprefixed base would let a bare `API_KEY` in any shell populate a service
credential, which is exactly the cross-configuration AD-025 separated `KB_CLI__`
from `KB_API__` to prevent.

`api_key` carries no default, so a missing one fails when settings are
constructed rather than at the first request.
"""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict

from platform_core import BaseServiceSettings

__all__ = ["KbClientSettings"]


class KbClientSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_CLIENT__")

    base_url: str = "http://localhost:8000"
    """The service root, without `/v1`.

    Locally this is the port `just run` binds. Against production it is the
    tailnet address `tailscale serve` publishes — AD-023 leaves no other way in,
    which is why this defaults to the development value rather than a hostname
    that only resolves from a tailnet-joined machine.
    """

    api_key: SecretStr

    provider: str | None = None
    """Which embedding provider's collection to work against (AD-006).

    `None` sends no `provider` field at all and lets the service apply its own
    default, which is one fewer place for the two to disagree.
    """

    user_agent: str | None = None
    """What this consumer calls itself, sent on every request.

    `platform-fastapi`'s audit middleware records it on the tier-1 row, so this
    is what makes MCP traffic separable from CLI traffic in
    `kb_audit.request_logs` — both hold their own key, but a `key_id` says which
    credential was used and not which program used it. `None` leaves httpx's
    default, which is what an unconfigured consumer deserves.
    """

    timeout_seconds: float = Field(default=30.0, gt=0)

    ingest_timeout_seconds: float = Field(default=300.0, gt=0)
    """Ingest gets its own budget because it is not the same shape of request.

    A search is one embedding call; ingesting a large document is a chunked
    embed of the whole thing, and the corpus averages ~0.5 MB a file. One
    timeout covering both would be either too tight for ingest or useless for
    search.
    """

    @field_validator("base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        # `httpx.AsyncClient(base_url=...)` joins paths differently depending on
        # whether the base ends in a slash. Normalising here means a config that
        # says `http://kb-api:8000/` and one that does not build the same URL.
        return value.rstrip("/")
