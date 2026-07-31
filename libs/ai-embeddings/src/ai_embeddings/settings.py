"""Provider configuration and the one HTTP client each provider gets.

Timeouts are Design §5's: 5 s connect, 30 s read. They are explicit because an
embedding call is the slowest thing in a search (200-400 ms typical, unbounded
when the provider is having a bad day) and it sits inside a 1.5 s p95 target.

The client is built here and passed into the adapter rather than created by it,
so a service opens one per provider in its lifespan handler and reuses the
connection pool. Per-request clients pay a TLS handshake on every search.
"""

import httpx
from pydantic import BaseModel, Field, SecretStr

__all__ = ["EmbeddingProviderSettings", "create_http_client"]


class EmbeddingProviderSettings(BaseModel):
    """Nested under a service's settings, as `KB_API__OPENAI__API_KEY`."""

    model_config = {"frozen": True}

    # No default. A missing key fails at startup rather than on the first
    # search, which is the whole point of loading settings eagerly.
    api_key: SecretStr

    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    read_timeout_seconds: float = Field(default=30.0, gt=0)

    # Bounded retry with jittered backoff on 429/5xx, delegated to the provider
    # SDK, which already implements exactly that. Ingest is retryable by design
    # (AD-008), so this stays small rather than heroic.
    max_retries: int = Field(default=2, ge=0)

    # Overrides the driver's default model. Changing it is a re-embed
    # migration, not a config flip (AD-006).
    model_id: str | None = None
    dimensions: int | None = None


def create_http_client(settings: EmbeddingProviderSettings) -> httpx.AsyncClient:
    """The shared client for one provider. Close it in the lifespan handler."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.read_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        ),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
