"""`StatsRepository` against real Postgres.

The test that matters is the burst count. The first implementation wrapped
`platform_db.repeat_bursts()` — a row query carrying a `LIMIT` — in a `count()`,
so the answer saturated at the limit: a runaway loop of four thousand identical
searches reported as twenty. Every assertion below that seeded fewer rows than
the cap would have passed, which is why this one seeds more.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from kb_api.adapters.postgres.stats import StatsRepository
from platform_db import AuditRecord, Database, Outcome, request_logs

pytestmark = pytest.mark.integration

BURSTS = 50
"""Deliberately more than any limit the old implementation used."""


@pytest.fixture
def repository() -> StatsRepository:
    return StatsRepository()


async def _seed_bursts(database: Database, count: int, *, age_days: float = 0.0) -> None:
    created = datetime.now(UTC) - timedelta(days=age_days)
    records = [
        AuditRecord(
            request_id=uuid4(),
            method="POST",
            path="/v1/search",
            status_code=200,
            outcome=Outcome.SUCCESS,
            latency_ms=12,
            key_id="n8n",
            repeat_burst=True,
            created_at=created,
        )
        for _ in range(count)
    ]
    async with database.session() as session:
        await session.execute(request_logs.insert(), [record.to_row() for record in records])


@pytest.fixture
async def seeded(database: Database) -> AsyncIterator[Database]:
    yield database


async def test_the_burst_count_does_not_saturate(
    seeded: Database, repository: StatsRepository
) -> None:
    """The regression. A capped count under-reports the incident it exists for."""
    await _seed_bursts(seeded, BURSTS)

    async with seeded.session() as session:
        assert await repository.recent_bursts(session) == BURSTS


async def test_unflagged_requests_are_not_counted(
    seeded: Database, repository: StatsRepository
) -> None:
    await _seed_bursts(seeded, 3)
    async with seeded.session() as session:
        await session.execute(
            request_logs.insert(),
            [
                AuditRecord(
                    request_id=uuid4(),
                    method="POST",
                    path="/v1/search",
                    status_code=200,
                    outcome=Outcome.SUCCESS,
                    latency_ms=5,
                ).to_row()
                for _ in range(20)
            ],
        )

    async with seeded.session() as session:
        assert await repository.recent_bursts(session) == 3


async def test_the_window_excludes_older_bursts(
    seeded: Database, repository: StatsRepository
) -> None:
    """Last week's incident is not this week's."""
    await _seed_bursts(seeded, 4)
    await _seed_bursts(seeded, 9, age_days=30)

    async with seeded.session() as session:
        assert await repository.recent_bursts(session, days=7) == 4
        assert await repository.recent_bursts(session, days=60) == 13


async def test_an_empty_trail_counts_zero(seeded: Database, repository: StatsRepository) -> None:
    async with seeded.session() as session:
        assert await repository.recent_bursts(session) == 0
