"""Configuration for everything that talks to Postgres.

`postgres` is the nested model `platform-db` publishes, mounted rather than
restated, so `KB_API__POSTGRES__DSN` means the same thing in every service that
mounts it (Design §5).

Deliberately *not* built on `HttpServiceSettings`. Migrations run as their own
process — from an init container at deploy time (Phase 9) — and that process has
a DSN but no API keys. Inheriting the HTTP settings would make `alembic upgrade
head` fail on a missing `KB_API__API_KEYS`, which is a startup dependency the
migration genuinely does not have. The service's own settings class arrives with
the composition root in Phase 6 and mounts the HTTP fields there.
"""

from functools import lru_cache

from pydantic_settings import SettingsConfigDict

from platform_core import BaseServiceSettings
from platform_db import DatabaseSettings

__all__ = ["DatabaseConfig", "get_database_config"]


class DatabaseConfig(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_API__")

    postgres: DatabaseSettings


@lru_cache(maxsize=1)
def get_database_config() -> DatabaseConfig:
    """Read the environment once, so a misconfiguration is reported once."""
    return DatabaseConfig()
