"""The Phase 4 exit criterion: the migration applies and rolls back cleanly.

Its own module, so it gets its own container from the module-scoped fixtures —
rolling the schema back underneath the repository tests would be an interesting
way to fail them.
"""

import asyncio
import subprocess
from collections.abc import Callable

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.integration

Alembic = Callable[..., subprocess.CompletedProcess[str]]

EXPECTED_TABLES = {
    ("kb", "documents"),
    ("kb", "chunks"),
    ("kb_audit", "request_logs"),
    ("kb_audit", "token_usage_logs"),
    ("kb_audit", "ingest_logs"),
    ("kb_audit", "error_logs"),
}


async def _tables(dsn: str) -> set[tuple[str, str]]:
    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                text(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema IN ('kb', 'kb_audit')"
                )
            )
            return {(str(schema), str(name)) for schema, name in rows.all()}
    finally:
        await engine.dispose()


def test_the_migration_applies_to_a_clean_database(alembic: Alembic, dsn: str) -> None:
    """Both schemas, every table, on a database that had none of it.

    `alembic_version` lands in `kb` rather than `public`, which is what stops a
    second service on the same instance from overwriting this one's migration
    state.
    """
    alembic("upgrade", "head")

    tables = asyncio.run(_tables(dsn))

    assert tables >= EXPECTED_TABLES
    assert ("kb", "alembic_version") in tables


def test_the_migration_rolls_back_cleanly(alembic: Alembic, dsn: str) -> None:
    """Down to base and back up again.

    Rolling back once proves the `downgrade` body runs; going back up proves it
    left the database in a state `upgrade` can be applied to again. A downgrade
    that drops a table but leaves a constraint or type behind passes the first
    check and fails the second, which is the failure worth catching.
    """
    alembic("upgrade", "head")

    alembic("downgrade", "base")
    after_downgrade = asyncio.run(_tables(dsn))

    alembic("upgrade", "head")
    after_reupgrade = asyncio.run(_tables(dsn))

    # `alembic_version` survives by design — it is the record of the rollback.
    assert after_downgrade == {("kb", "alembic_version")}
    assert after_reupgrade >= EXPECTED_TABLES


def test_the_models_have_not_drifted_from_the_migration(alembic: Alembic) -> None:
    """`alembic check` — what CI runs to catch a model edited without a revision.

    Also the only thing exercising the autogenerate comparison end to end,
    including that the GIN operator class is expressed in a form Alembic can
    compare rather than one it warns about and skips.
    """
    alembic("upgrade", "head")

    result = alembic("check")

    assert "No new upgrade operations detected" in result.stdout
