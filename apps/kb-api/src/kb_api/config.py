"""Configuration, in two classes for one reason.

`DatabaseConfig` carries only what a Postgres connection needs. Migrations run
as their own process — from an init container at deploy time (Phase 9) — and
that process has a DSN but no API keys. If Alembic loaded the full service
settings, `alembic upgrade head` would fail on a missing `KB_API__API_KEYS`,
which is a startup dependency the migration genuinely does not have.

`KbApiSettings` is the service's own, mounting `HttpServiceSettings` for keys,
CORS, health budget, and body cap, plus the nested models `platform-db` and
`ai-embeddings` publish. Nesting rather than restating is what makes
`KB_API__POSTGRES__DSN` and `KB_API__OPENAI__API_KEY` mean the same thing in
every service that mounts them (Design §5).

Both provider blocks default to `None`. A provider is configured or it is not,
and the registry is built from whichever ones are — so running with only an
OpenAI key is a supported deployment rather than a half-broken one.
`default_provider` then has to name one that exists, which the composition root
checks at startup where it is cheap to fix.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ai_embeddings import EmbeddingProviderSettings
from platform_core import BaseServiceSettings
from platform_db import AuditSettings, DatabaseSettings
from platform_fastapi import HttpServiceSettings

__all__ = [
    "ChromaSettings",
    "DatabaseConfig",
    "KbApiSettings",
    "get_database_config",
    "get_settings",
]


class DatabaseConfig(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_API__")

    postgres: DatabaseSettings


class ChromaSettings(BaseServiceSettings):
    """Where the vector store is — `KB_API__CHROMA__HOST`, and so on.

    Tenant and database are Chroma's own namespacing. They are settings rather
    than constants because one Chroma server hosting a second service is a
    realistic deployment, and discovering that after the fact is a data
    migration rather than a config change.
    """

    model_config = SettingsConfigDict(env_prefix="KB_API__CHROMA__")

    host: str = "localhost"
    port: int = Field(default=8000, gt=0, lt=65536)
    ssl: bool = False
    tenant: str = "default_tenant"
    database: str = "default_database"


class KbApiSettings(HttpServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_API__", env_nested_delimiter="__")

    service_name: str = "kb-api"
    service_version: str = "0.1.0"

    postgres: DatabaseSettings
    audit: AuditSettings = AuditSettings()
    chroma: ChromaSettings = ChromaSettings()

    openai: EmbeddingProviderSettings | None = None
    gemini: EmbeddingProviderSettings | None = None

    default_provider: str = "openai"

    # Startup reconciliation reads at most this many of each kind of stranded
    # document, so a pathological state cannot make readiness take unbounded
    # time. Whatever is left is picked up by the next restart.
    reconciliation_limit: int = Field(default=100, ge=1)

    # The query-embedding cache (AD-008, AD-021). Design §8 measured the hit
    # rate as near zero for the n8n workload, so these are sized to be free
    # rather than effective: 512 vectors at 1536 dimensions is roughly 3 MB.
    # Zero entries disables it.
    query_cache_size: int = Field(default=512, ge=0)
    query_cache_ttl_seconds: float = Field(default=900.0, gt=0)

    # AD-005's `$in` clause is built from this many document IDs at most. The
    # corpus is ~25 documents, so the cap exists for the pathological case, not
    # the expected one.
    tag_filter_limit: int = Field(default=2000, ge=1)


@lru_cache(maxsize=1)
def get_database_config() -> DatabaseConfig:
    """Read the environment once, so a misconfiguration is reported once."""
    return DatabaseConfig()


@lru_cache(maxsize=1)
def get_settings() -> KbApiSettings:
    return KbApiSettings()
