"""Tier 1: never drops a record, never fails a request (AD-013)."""

import asyncio

import pytest
from sqlalchemy.exc import OperationalError

from platform_db import AuditTrail, Outcome, SpillFile


@pytest.fixture
def trail(database, spill: SpillFile) -> AuditTrail:
    return AuditTrail(database, spill, reconcile_interval_seconds=0.02)


async def test_a_healthy_write_goes_straight_to_postgres(
    trail: AuditTrail, database, spill: SpillFile, make_record
) -> None:
    await trail.record(make_record())

    assert len(database.rows) == 1
    assert await spill.depth() == 0


async def test_the_row_carries_the_forensic_columns(
    trail: AuditTrail, database, make_record
) -> None:
    # These are the columns the trail exists for: who, from where, what was
    # asked, and how it ended (Design §2.2).
    await trail.record(
        make_record(
            key_id="n8n",
            client_ip="203.0.113.7",
            outcome=Outcome.AUTH_FAILED,
            status_code=401,
            error_code="unauthenticated",
            operation="search",
            payload={"query": "salary"},
        )
    )

    row = database.rows[0]
    assert row["key_id"] == "n8n"
    assert row["client_ip"] == "203.0.113.7"  # str, because asyncpg wants one for INET
    assert row["outcome"] == "auth_failed"
    assert row["error_code"] == "unauthenticated"
    assert row["payload"] == {"query": "salary"}


async def test_an_unreachable_postgres_spills_instead_of_dropping(
    trail: AuditTrail, database, spill: SpillFile, make_record
) -> None:
    database.available = False

    await trail.record(make_record())

    assert await spill.depth() == 1


async def test_recording_never_raises(trail: AuditTrail, database, make_record) -> None:
    # The request must not fail because the audit trail is unavailable. This is
    # the assertion that keeps a Postgres restart from bricking the service.
    database.available = False

    await trail.record(make_record())  # no pytest.raises: any exception fails here


async def test_a_record_survives_both_postgres_and_the_disk(
    trail: AuditTrail, database, spill: SpillFile, make_record, monkeypatch
) -> None:
    """Both durable surfaces gone: still no exception, and still logged.

    With nowhere to put the record, the structured log is the last place it can
    be recovered from, so it is emitted there in full rather than swallowed.
    """
    database.available = False

    async def unwritable(record) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spill, "append", unwritable)

    await trail.record(make_record())


async def test_the_reconciler_drains_the_spill_on_recovery(
    trail: AuditTrail, database, spill: SpillFile, make_records
) -> None:
    database.available = False
    for record in make_records(5):
        await trail.record(record)
    assert await trail.spill_depth() == 5

    database.available = True
    assert await trail.reconcile() == 5

    assert await trail.spill_depth() == 0
    assert len(database.rows) == 5


async def test_the_spilled_batch_reaches_postgres_in_one_insert(
    trail: AuditTrail, database, make_records
) -> None:
    database.available = False
    for record in make_records(20):
        await trail.record(record)

    database.available = True
    await trail.reconcile()

    # One executemany, not twenty round trips — recovery after a long outage
    # should not be its own load spike.
    assert len(database.session_obj.executed) == 1
    assert len(database.rows) == 20


async def test_reconciling_while_postgres_is_still_down_keeps_the_records(
    trail: AuditTrail, database, make_records
) -> None:
    database.available = False
    for record in make_records(3):
        await trail.record(record)

    with pytest.raises(OperationalError):
        await trail.reconcile()

    assert await trail.spill_depth() == 3


async def test_reconciling_an_empty_spill_costs_nothing(trail: AuditTrail, database) -> None:
    assert await trail.reconcile() == 0
    assert database.rows == []


async def test_the_background_reconciler_recovers_without_being_asked(
    trail: AuditTrail, database, make_records
) -> None:
    database.available = False
    for record in make_records(4):
        await trail.record(record)

    await trail.start()
    try:
        database.available = True
        async with asyncio.timeout(2):
            while await trail.spill_depth() > 0:
                await asyncio.sleep(0.01)
    finally:
        await trail.stop()

    assert len(database.rows) == 4


async def test_the_background_reconciler_survives_a_still_down_postgres(
    trail: AuditTrail, database, make_records
) -> None:
    # A reconciler that dies on the first failure means the spill is never
    # drained at all, which is the same as dropping the records.
    database.available = False
    for record in make_records(2):
        await trail.record(record)

    await trail.start()
    await asyncio.sleep(0.1)  # several failed passes
    database.available = True
    try:
        async with asyncio.timeout(2):
            while await trail.spill_depth() > 0:
                await asyncio.sleep(0.01)
    finally:
        await trail.stop()

    assert len(database.rows) == 2


async def test_start_and_stop_are_idempotent(trail: AuditTrail) -> None:
    await trail.stop()
    await trail.start()
    await trail.start()
    await trail.stop()
    await trail.stop()
