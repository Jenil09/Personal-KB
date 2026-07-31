"""Tier 2: bounded, batched, and honest about what it drops (AD-009, AD-013).

The exit criterion this suite exists for: a 10 000-record burst is absorbed
without unbounded memory growth, and saturation drops *predictably* — a known
count, not an approximate one.
"""

import asyncio

from sqlalchemy import Column, Integer, MetaData, Table, Text

from platform_db import AuditSettings, TelemetryEvent, TelemetrySink

_metadata = MetaData(schema="kb_audit")

token_usage_logs = Table(
    "token_usage_logs",
    _metadata,
    Column("request_id", Text),
    Column("tokens", Integer),
)

ingest_logs = Table(
    "ingest_logs",
    _metadata,
    Column("document_id", Text),
    Column("chunks", Integer),
)


def event(index: int = 0) -> TelemetryEvent:
    return TelemetryEvent(token_usage_logs, {"request_id": str(index), "tokens": index})


async def drain(sink: TelemetrySink) -> None:
    """Let the consumer catch up, without racing a fixed sleep."""
    async with asyncio.timeout(5):
        while sink.depth > 0:
            await asyncio.sleep(0.005)
    await sink.flush()


async def test_emitting_without_a_consumer_never_blocks(database, audit_settings) -> None:
    # The caller is mid-request. Enqueueing must not be able to wait on
    # anything, started or not.
    sink = TelemetrySink(database, audit_settings)

    for index in range(50):
        sink.emit(event(index))

    assert sink.depth == 50
    assert database.rows == []


async def test_a_started_sink_writes_what_it_was_given(database, audit_settings) -> None:
    sink = TelemetrySink(database, audit_settings)
    await sink.start()
    try:
        for index in range(25):
            sink.emit(event(index))
        await drain(sink)
    finally:
        await sink.stop()

    assert sink.written == 25
    assert sink.dropped == 0
    assert len(database.rows) == 25


async def test_records_are_batched_rather_than_inserted_one_by_one(
    database, audit_settings
) -> None:
    sink = TelemetrySink(database, audit_settings)
    await sink.start()
    try:
        for index in range(30):
            sink.emit(event(index))
        await drain(sink)
    finally:
        await sink.stop()

    # 30 records at a batch size of 10 is three inserts, not thirty. Batching
    # is the reason tier 2 is allowed to be high-volume at all.
    assert len(database.session_obj.executed) <= 5


async def test_a_full_queue_drops_a_countable_number(database, audit_settings) -> None:
    settings = audit_settings.model_copy(update={"telemetry_queue_size": 100})
    sink = TelemetrySink(database, settings)

    for index in range(130):
        sink.emit(event(index))

    # Exactly the overflow. "Roughly 30" would make the drop counter useless
    # for deciding whether an observability gap explains a missing record.
    assert sink.dropped == 30
    assert sink.depth == 100


async def test_a_ten_thousand_record_burst_stays_bounded(database) -> None:
    """The phase's stated exit criterion, asserted rather than assumed."""
    settings = AuditSettings(
        telemetry_queue_size=10_000,
        telemetry_batch_size=100,
        telemetry_flush_interval_seconds=0.01,
    )
    sink = TelemetrySink(database, settings)
    await sink.start()
    try:
        for index in range(10_000):
            sink.emit(event(index))
            # Memory ceiling is the queue bound, and it holds throughout the
            # burst rather than only at the end.
            assert sink.depth <= 10_000
        await drain(sink)
    finally:
        await sink.stop()

    assert sink.written == 10_000
    assert sink.dropped == 0


async def test_saturation_past_the_bound_drops_the_excess(database) -> None:
    settings = AuditSettings(telemetry_queue_size=10_000, telemetry_batch_size=100)
    sink = TelemetrySink(database, settings)  # deliberately not started

    for index in range(12_500):
        sink.emit(event(index))

    assert sink.depth == 10_000
    assert sink.dropped == 2_500


async def test_a_failing_postgres_costs_telemetry_but_not_the_process(
    database, audit_settings
) -> None:
    # The tier-2 contract: these are droppable. Tier 1 is the one that is not,
    # and it does not come through here.
    database.available = False
    sink = TelemetrySink(database, audit_settings)
    await sink.start()
    try:
        for index in range(10):
            sink.emit(event(index))
        await drain(sink)
    finally:
        await sink.stop()

    assert sink.dropped == 10
    assert sink.written == 0


async def test_a_batch_spanning_tables_inserts_into_each(database, audit_settings) -> None:
    sink = TelemetrySink(database, audit_settings)
    await sink.start()
    try:
        sink.emit(TelemetryEvent(token_usage_logs, {"request_id": "a", "tokens": 1}))
        sink.emit(TelemetryEvent(ingest_logs, {"document_id": "b", "chunks": 2}))
        await drain(sink)
    finally:
        await sink.stop()

    assert sink.written == 2
    assert len(database.session_obj.executed) == 2


async def test_stopping_flushes_what_is_queued(database, audit_settings) -> None:
    sink = TelemetrySink(database, audit_settings)
    await sink.start()
    for index in range(40):
        sink.emit(event(index))

    await sink.stop()

    # Shutdown is the one moment tier 2 gets a real chance to not lose things.
    assert sink.written == 40
    assert sink.dropped == 0


async def test_a_wedged_drain_is_abandoned_rather_than_hanging_shutdown(
    database, audit_settings
) -> None:
    settings = audit_settings.model_copy(update={"telemetry_drain_timeout_seconds": 0.05})
    sink = TelemetrySink(database, settings)
    # Never started, so nothing will ever drain it: stands in for a Postgres
    # that has stopped answering.
    for index in range(5):
        sink.emit(event(index))

    await sink.flush()

    assert sink.dropped == 5


async def test_start_and_stop_are_idempotent(database, audit_settings) -> None:
    sink = TelemetrySink(database, audit_settings)

    await sink.stop()
    await sink.start()
    await sink.start()
    await sink.stop()
    await sink.stop()
