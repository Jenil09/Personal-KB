"""Engine construction and the health probe.

Session transaction semantics need a real database and live in
`test_audit_integration.py`; what is checkable without one is that the settings
actually reach the pool, and that an unreachable Postgres reports rather than
raises.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.pool import QueuePool

from platform_db import Database, DatabaseSettings

UNREACHABLE = "postgresql+asyncpg://kb:kb@127.0.0.1:1/kb"


@pytest.fixture
async def unreachable() -> AsyncIterator[Database]:
    database = Database(DatabaseSettings(dsn=UNREACHABLE, connect_timeout_seconds=1.0))
    yield database
    await database.dispose()


def test_pool_settings_reach_the_engine() -> None:
    settings = DatabaseSettings(dsn=UNREACHABLE, pool_size=3, max_overflow=7)

    pool = Database(settings).engine.pool

    # QueuePool's own accessors, reached through the base `Pool` type mypy sees
    # on `engine.pool`. The overflow bound is the point of configuring any of
    # this: 2 vCPU cannot execute thirty concurrent statements however many
    # connections the pool is willing to open.
    assert isinstance(pool, QueuePool)
    assert pool.size() == 3
    assert pool._max_overflow == 7


def test_constructing_a_database_does_not_connect() -> None:
    # Settings are built at import time in the composition root; a constructor
    # that dialled out would make an unreachable database an import error.
    Database(DatabaseSettings(dsn=UNREACHABLE))


async def test_ping_reports_rather_than_raises(unreachable: Database) -> None:
    # This backs `/health`. A probe that raises turns a degraded dependency
    # into a 500 on the endpoint whose job is to describe the degradation.
    assert await unreachable.ping() is False


async def test_dispose_is_safe_to_repeat(unreachable: Database) -> None:
    await unreachable.dispose()
    await unreachable.dispose()
