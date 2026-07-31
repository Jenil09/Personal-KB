"""Configuration that fails at startup rather than at first query."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from platform_db import AuditSettings, DatabaseSettings

DSN = "postgresql+asyncpg://kb:kb@localhost:5433/kb"


def test_dsn_is_required() -> None:
    with pytest.raises(ValidationError, match="dsn"):
        DatabaseSettings()  # type: ignore[call-arg]


def test_a_sync_dsn_is_rejected() -> None:
    # psycopg2 behind an async engine blocks the event loop on every query and
    # looks like a performance problem, so it has to fail here instead.
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        DatabaseSettings(dsn="postgresql://kb:kb@localhost:5433/kb")


def test_the_dsn_is_a_secret() -> None:
    settings = DatabaseSettings(dsn=DSN)

    assert "kb:kb" not in repr(settings)
    assert settings.dsn.get_secret_value() == DSN


def test_pool_defaults_are_sized_for_one_worker_on_two_vcpu() -> None:
    settings = DatabaseSettings(dsn=DSN)

    assert settings.pool_size + settings.max_overflow == 10


@pytest.mark.parametrize(
    "field, value",
    [
        ("pool_size", 0),
        ("max_overflow", -1),
        ("pool_timeout_seconds", 0),
        ("connect_timeout_seconds", 0),
        ("statement_timeout_seconds", -1),
    ],
)
def test_nonsensical_pool_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        DatabaseSettings(dsn=DSN, **{field: value})


def test_settings_are_frozen() -> None:
    settings = DatabaseSettings(dsn=DSN)

    with pytest.raises(ValidationError):
        settings.pool_size = 20  # type: ignore[misc]


def test_audit_defaults_match_the_decision() -> None:
    settings = AuditSettings()

    # AD-009's bounded queue and batch, retained by AD-013.
    assert settings.telemetry_queue_size == 10_000
    assert settings.telemetry_batch_size == 100
    assert settings.telemetry_flush_interval_seconds == 0.5
    assert isinstance(settings.spill_path, Path)


def test_an_empty_telemetry_queue_is_rejected() -> None:
    # maxsize=0 in asyncio means unbounded, which is the one thing the drop
    # policy exists to prevent.
    with pytest.raises(ValidationError, match="telemetry_queue_size"):
        AuditSettings(telemetry_queue_size=0)
