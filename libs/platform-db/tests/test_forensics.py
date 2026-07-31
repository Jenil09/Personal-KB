"""The forensic query set: right rows, and reached through the right index.

Correctness and index usage are asserted together against a seeded table,
because the two failures look identical from the outside — a query that answers
correctly by reading every row is the one that will be run during an incident,
on the largest the table has ever been, by someone who needs an answer now.

Seeded well past the real table size for the same reason as the tag index tests:
Postgres scans a small table sequentially however good the index is, and it is
right to. The volume here is roughly a year of traffic at AD-014's design target.
"""

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

from platform_db import (
    Database,
    DatabaseSettings,
    Outcome,
    activity_by_ip,
    activity_by_key,
    audit_metadata,
    failures_in_window,
    ingests_by_key,
    repeat_bursts,
    request_logs,
    traffic_summary,
)
from platform_db.testing import plan_for, uses_index

pytestmark = pytest.mark.integration

IMAGE = "docker.io/library/postgres:16-alpine"

# ~a year at the 200/day design target (Design §8), so the planner has a real
# choice and the selective predicates below are genuinely selective.
SEEDED = 40_000

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

KEYS = ("n8n", "cli", "manual")
NEEDLE_KEY = "compromised"
NEEDLE_IP = "203.0.113.99"
BUSY_IP = "198.51.100.10"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def port() -> int:
    return _free_port()


@pytest.fixture(scope="module")
def postgres(port: int) -> Iterator[PostgresContainer]:
    container = PostgresContainer(IMAGE).with_bind_ports(5432, port)
    container.start()
    try:
        yield container
    finally:
        # Fixtures own teardown: Ryuk is disabled under rootless Podman (AD-015).
        with suppress(Exception):
            container.stop()


@pytest.fixture(scope="module")
def seeded(postgres: PostgresContainer, port: int) -> str:
    """One seeded database for the module. Every test here only reads."""
    dsn = f"postgresql+asyncpg://test:test@127.0.0.1:{port}/test"
    asyncio.run(_seed(dsn))
    return dsn


@pytest.fixture
async def database(seeded: str) -> AsyncIterator[Database]:
    instance = Database(DatabaseSettings(dsn=seeded, connect_timeout_seconds=5.0))
    try:
        yield instance
    finally:
        await instance.dispose()


def _rows() -> list[dict[str, object]]:
    """Deterministic traffic, with a small number of planted needles.

    The needles are rare on purpose: a key responsible for a third of the table
    is one the planner should answer with a sequential scan, so a query about it
    would prove nothing about the index.
    """
    rows: list[dict[str, object]] = []
    for index in range(SEEDED):
        outcome = Outcome.SUCCESS
        status = 200
        if index % 997 == 0:
            outcome, status = Outcome.AUTH_FAILED, 401
        elif index % 499 == 0:
            outcome, status = Outcome.RATE_LIMITED, 429
        elif index % 331 == 0:
            outcome, status = Outcome.SERVER_ERROR, 500

        rows.append(
            {
                "request_id": uuid4(),
                "key_id": KEYS[index % len(KEYS)],
                "client_ip": BUSY_IP,
                "user_agent": "n8n/1.0",
                "method": "POST",
                "path": "/v1/search",
                "status_code": status,
                "outcome": outcome.value,
                "error_code": None if outcome is Outcome.SUCCESS else outcome.value,
                "latency_ms": 40 + (index % 60),
                "operation": "search",
                "payload": None,
                "repeat_burst": index % 5_000 == 0,
                # Spread backwards from NOW so window queries have something to
                # cut. One row per ~13 minutes at this volume.
                "created_at": NOW - timedelta(minutes=index * 13),
            }
        )

    # The needles: one key and one address worth investigating, with a mix of
    # searches and ingests so `ingests_by_key` has something to narrow.
    for index in range(20):
        rows.append(
            {
                "request_id": uuid4(),
                "key_id": NEEDLE_KEY,
                "client_ip": NEEDLE_IP,
                "user_agent": "curl/8.0",
                "method": "POST",
                "path": "/v1/documents" if index % 2 else "/v1/search",
                "status_code": 200,
                "outcome": Outcome.SUCCESS.value,
                "error_code": None,
                "latency_ms": 100 + index,
                "operation": "ingest" if index % 2 else "search",
                "payload": None,
                "repeat_burst": False,
                "created_at": NOW - timedelta(hours=index),
            }
        )
    return rows


async def _seed(dsn: str) -> None:
    database = Database(DatabaseSettings(dsn=dsn, connect_timeout_seconds=5.0))
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text("CREATE SCHEMA IF NOT EXISTS kb_audit"))
            await connection.run_sync(audit_metadata.create_all)
            await connection.execute(request_logs.insert(), _rows())
            # Without statistics the planner works from defaults, and its
            # choices say nothing about the data actually present.
            await connection.execute(text("ANALYZE kb_audit.request_logs"))
    finally:
        await database.dispose()


async def test_activity_by_key_returns_only_that_key(database: Database) -> None:
    """The first question asked after a key is suspected."""
    async with database.session() as session:
        rows = (await session.execute(activity_by_key(NEEDLE_KEY))).all()

    assert len(rows) == 20
    assert {row.key_id for row in rows} == {NEEDLE_KEY}
    # Newest first: an investigation starts at the most recent activity.
    assert [row.created_at for row in rows] == sorted(
        (row.created_at for row in rows), reverse=True
    )


async def test_activity_by_key_uses_its_index(database: Database) -> None:
    async with database.session() as session:
        plan = await plan_for(session, activity_by_key(NEEDLE_KEY))

    assert uses_index(plan, "ix_request_logs_key_id_created_at"), plan


async def test_activity_by_key_respects_a_window(database: Database) -> None:
    async with database.session() as session:
        rows = (
            await session.execute(activity_by_key(NEEDLE_KEY, since=NOW - timedelta(hours=5)))
        ).all()

    # Half-open `[since, until)` — hours 0 through 5 inclusive of the boundary.
    assert len(rows) == 6


async def test_a_window_is_half_open_so_consecutive_windows_tile(
    database: Database,
) -> None:
    """The boundary row belongs to exactly one window, not both or neither."""
    boundary = NOW - timedelta(hours=10)

    async with database.session() as session:
        earlier = (await session.execute(activity_by_key(NEEDLE_KEY, until=boundary))).all()
        later = (await session.execute(activity_by_key(NEEDLE_KEY, since=boundary))).all()
        whole = (await session.execute(activity_by_key(NEEDLE_KEY))).all()

    assert len(earlier) + len(later) == len(whole)
    assert not {row.id for row in earlier} & {row.id for row in later}


async def test_activity_by_ip_crosses_keys(database: Database) -> None:
    """Scoped to the address on purpose — one source using several keys is the point."""
    async with database.session() as session:
        rows = (await session.execute(activity_by_ip(NEEDLE_IP))).all()

    assert len(rows) == 20
    assert {str(row.client_ip) for row in rows} == {NEEDLE_IP}


async def test_activity_by_ip_uses_its_index(database: Database) -> None:
    async with database.session() as session:
        plan = await plan_for(session, activity_by_ip(NEEDLE_IP))

    assert uses_index(plan, "ix_request_logs_client_ip_created_at"), plan


async def test_failures_excludes_successes(database: Database) -> None:
    async with database.session() as session:
        rows = (await session.execute(failures_in_window(limit=100_000))).all()

    assert rows
    assert Outcome.SUCCESS.value not in {row.outcome for row in rows}


async def test_failures_uses_the_partial_index(database: Database) -> None:
    """The partial index holds only non-success rows, so it is a fraction of the table."""
    async with database.session() as session:
        plan = await plan_for(session, failures_in_window())

    assert uses_index(plan, "ix_request_logs_outcome"), plan


async def test_failures_can_be_narrowed_to_one_outcome(database: Database) -> None:
    async with database.session() as session:
        rows = (
            await session.execute(
                failures_in_window(outcomes=(Outcome.AUTH_FAILED,), limit=100_000)
            )
        ).all()

    assert rows
    assert {row.outcome for row in rows} == {Outcome.AUTH_FAILED.value}


async def test_repeat_bursts_returns_only_flagged_rows(database: Database) -> None:
    """AD-014's sharpest signal: the same request over and over, not merely a lot of them."""
    async with database.session() as session:
        rows = (await session.execute(repeat_bursts(limit=100_000))).all()

    assert rows
    assert all(row.repeat_burst for row in rows)


async def test_repeat_bursts_uses_the_partial_index(database: Database) -> None:
    async with database.session() as session:
        plan = await plan_for(session, repeat_bursts())

    assert uses_index(plan, "ix_request_logs_repeat_burst"), plan


async def test_ingests_by_key_narrows_to_the_ingest_operation(database: Database) -> None:
    """AD-014's blast-radius query: everything this key ever put in."""
    async with database.session() as session:
        rows = (await session.execute(ingests_by_key(NEEDLE_KEY))).all()

    assert len(rows) == 10
    assert {row.operation for row in rows} == {"ingest"}
    assert {row.key_id for row in rows} == {NEEDLE_KEY}


async def test_ingests_by_key_uses_the_key_index(database: Database) -> None:
    async with database.session() as session:
        plan = await plan_for(session, ingests_by_key(NEEDLE_KEY))

    assert uses_index(plan, "ix_request_logs_key_id_created_at"), plan


async def test_traffic_summary_groups_by_key_and_outcome(database: Database) -> None:
    """The aggregate AD-014's thresholds are meant to be tuned against."""
    async with database.session() as session:
        rows = (await session.execute(traffic_summary(since=NOW - timedelta(days=30)))).all()

    assert rows
    pairs = [(row.key_id, row.outcome) for row in rows]
    assert len(pairs) == len(set(pairs))
    assert all(row.requests > 0 for row in rows)
    assert all(row.first_seen <= row.last_seen for row in rows)


async def test_traffic_summary_uses_the_created_at_index_for_its_window(
    database: Database,
) -> None:
    """No selective predicate, so the window itself has to be what bounds the scan."""
    async with database.session() as session:
        plan = await plan_for(session, traffic_summary(since=NOW - timedelta(days=2)))

    assert uses_index(plan, "ix_request_logs_created_at"), plan


async def test_every_query_is_bounded(database: Database) -> None:
    """An unbounded scan during an incident is how it becomes the second outage."""
    async with database.session() as session:
        rows = (await session.execute(activity_by_key("n8n", limit=5))).all()

    assert len(rows) == 5
