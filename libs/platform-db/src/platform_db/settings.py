"""Database and audit configuration.

Both are nested models rather than settings classes of their own: a service
mounts them on its own `BaseServiceSettings` subclass, so they populate from
`KB_API__POSTGRES__DSN` and `KB_API__AUDIT__SPILL_PATH` without each service
inventing its own spelling.

Defaults are sized for the deployment AD-015 describes — 2 vCPU, 4 GB, a single
Uvicorn worker sharing the box with Postgres and Chroma. One worker means one
event loop, so the pool bounds concurrent statements rather than concurrent
processes; ten connections is already more than 2 vCPU can execute at once.
"""

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator

__all__ = ["AuditSettings", "DatabaseSettings"]

_ASYNC_DRIVER = "postgresql+asyncpg"


class DatabaseSettings(BaseModel):
    model_config = {"frozen": True}

    # No default: a missing DSN must fail at startup, not at first query.
    dsn: SecretStr

    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    # Below any sensible idle-connection reaper, so the pool never hands out a
    # connection the server has already closed.
    pool_recycle_seconds: int = Field(default=1800, gt=0)

    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    statement_timeout_seconds: float = Field(default=10.0, gt=0)

    echo: bool = False

    @field_validator("dsn")
    @classmethod
    def _require_async_driver(cls, value: SecretStr) -> SecretStr:
        # A `postgresql://` DSN loads psycopg2 and blocks the event loop on
        # every query. It fails late and looks like a performance problem
        # rather than a configuration one, so reject it at startup.
        if not value.get_secret_value().startswith(f"{_ASYNC_DRIVER}://"):
            raise ValueError(f"must be an async DSN beginning {_ASYNC_DRIVER}://")
        return value


class AuditSettings(BaseModel):
    """Tier-1 durability and tier-2 throughput knobs (AD-013)."""

    model_config = {"frozen": True}

    # Tier 1 — where records go when Postgres is unreachable. On a volume that
    # survives a container restart, or the guarantee is not a guarantee.
    spill_path: Path = Path("data/audit.spill.jsonl")

    # Tier 2 — AD-009's bounded queue and drop policy, retained by AD-013.
    telemetry_queue_size: int = Field(default=10_000, ge=1)
    telemetry_batch_size: int = Field(default=100, ge=1)
    telemetry_flush_interval_seconds: float = Field(default=0.5, gt=0)
    # A drain that outlives this on shutdown is abandoned; tier 2 is
    # best-effort, and a hung Postgres must not block the process from exiting.
    telemetry_drain_timeout_seconds: float = Field(default=5.0, gt=0)
