"""The spill file, which is the whole of AD-013's never-dropped guarantee.

Every case here is a way records get lost in implementations that look correct:
a drain that truncates instead of rotating, a crash between reading and
deleting, a torn line that fails the whole file, a rename racing an append.
"""

import asyncio

import pytest

from platform_db import AuditRecord, SpillFile


async def test_draining_an_absent_file_is_a_no_op(spill: SpillFile) -> None:
    async def handler(batch) -> None:
        raise AssertionError("handler must not be called with nothing to drain")

    assert await spill.drain(handler) == 0
    assert await spill.depth() == 0


async def test_append_creates_the_directory(spill: SpillFile, make_record) -> None:
    # The spill path points at a volume the operator configured; the service
    # must not need someone to mkdir it first.
    assert not spill.path.parent.exists()

    await spill.append(make_record())

    assert spill.path.exists()
    assert await spill.depth() == 1


async def test_records_survive_a_round_trip_intact(spill: SpillFile, make_record) -> None:
    original = make_record(
        client_ip="203.0.113.7",
        user_agent="n8n/1.0",
        operation="search",
        payload={"query": "vector databases", "filters": {"type": "note"}},
        repeat_burst=True,
    )
    await spill.append(original)

    drained: list[AuditRecord] = []

    async def handler(batch) -> None:
        drained.extend(batch)

    assert await spill.drain(handler) == 1
    assert drained == [original]


async def test_created_at_survives_the_delay(spill: SpillFile, make_record) -> None:
    # The point of stamping the time at the request rather than at the insert:
    # a record drained an hour later must not claim to have happened then.
    original = make_record()
    await spill.append(original)

    drained: list[AuditRecord] = []

    async def handler(batch) -> None:
        drained.extend(batch)

    await spill.drain(handler)

    assert drained[0].created_at == original.created_at


async def test_a_successful_drain_empties_the_file(spill: SpillFile, make_records) -> None:
    for record in make_records(5):
        await spill.append(record)
    assert await spill.depth() == 5

    async def handler(batch) -> None:
        return None

    assert await spill.drain(handler) == 5
    assert await spill.depth() == 0
    assert not spill.path.exists()


async def test_a_failing_handler_keeps_every_record(spill: SpillFile, make_records) -> None:
    for record in make_records(3):
        await spill.append(record)

    async def failing(batch) -> None:
        raise ConnectionError("postgres is still down")

    with pytest.raises(ConnectionError):
        await spill.drain(failing)

    # Not "most of them" — the caller retries, so the count has to be exact.
    assert await spill.depth() == 3


async def test_a_retry_after_a_failed_drain_gets_the_same_records(
    spill: SpillFile, make_records
) -> None:
    written = make_records(4)
    for record in written:
        await spill.append(record)

    async def failing(batch) -> None:
        raise ConnectionError("postgres is still down")

    with pytest.raises(ConnectionError):
        await spill.drain(failing)

    drained: list[AuditRecord] = []

    async def handler(batch) -> None:
        drained.extend(batch)

    assert await spill.drain(handler) == 4
    assert drained == list(written)


async def test_a_crash_mid_drain_leaves_the_records_recoverable(
    spill: SpillFile, make_records
) -> None:
    """The rotation is what makes this survivable.

    Simulates a process dying after the rename but before the delete: the
    `.draining` file is still on disk, and the next drain must find it rather
    than walk past it.
    """
    for record in make_records(3):
        await spill.append(record)

    async def crash(batch) -> None:
        raise KeyboardInterrupt("SIGINT during the insert")

    with pytest.raises(KeyboardInterrupt):
        await spill.drain(crash)

    draining = spill.path.with_name(spill.path.name + ".draining")
    assert draining.exists()

    recovered: list[AuditRecord] = []

    async def handler(batch) -> None:
        recovered.extend(batch)

    assert await spill.drain(handler) == 3
    assert await spill.depth() == 0


async def test_records_written_during_a_drain_are_not_lost(
    spill: SpillFile, make_record, make_records
) -> None:
    for record in make_records(2):
        await spill.append(record)

    late = make_record(latency_ms=999)
    drained: list[AuditRecord] = []

    async def handler(batch) -> None:
        # Postgres came back and the service is serving again; a request that
        # fails to insert while this batch is in flight appends behind it.
        await spill.append(late)
        drained.extend(batch)

    assert await spill.drain(handler) == 2
    assert await spill.depth() == 1

    remaining: list[AuditRecord] = []

    async def collect_remaining(batch) -> None:
        remaining.extend(batch)

    await spill.drain(collect_remaining)
    assert remaining == [late]


async def test_a_torn_line_does_not_strand_the_rest_of_the_file(
    spill: SpillFile, make_records
) -> None:
    # fsync rules out most corruption but not a write interrupted by power
    # loss. Failing the drain on it would strand every record behind it.
    for record in make_records(2):
        await spill.append(record)
    with spill.path.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "0f9d3e9c-6d2e-4a1c-9f1')

    drained: list[AuditRecord] = []

    async def handler(batch) -> None:
        drained.extend(batch)

    assert await spill.drain(handler) == 2
    assert await spill.depth() == 0


async def test_depth_counts_a_batch_that_is_mid_drain(spill: SpillFile, make_records) -> None:
    # `/health` reports degraded off this number. A batch in flight is still
    # unreconciled, so hiding it would clear the warning too early.
    for record in make_records(3):
        await spill.append(record)

    async def failing(batch) -> None:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await spill.drain(failing)

    assert await spill.depth() == 3


async def test_concurrent_appends_all_land(spill: SpillFile, make_records) -> None:
    records = make_records(50)

    await asyncio.gather(*(spill.append(record) for record in records))

    assert await spill.depth() == 50


async def test_an_append_racing_a_drain_is_never_orphaned(spill: SpillFile, make_records) -> None:
    """The lock's reason for existing.

    Without it a rename can land between an append opening the file and writing
    to it, and that record ends up in a `.draining` file already read past.
    """
    first = make_records(20)
    for record in first:
        await spill.append(record)

    drained: list[AuditRecord] = []

    async def handler(batch) -> None:
        drained.extend(batch)

    second = make_records(20)
    drain_task = asyncio.create_task(spill.drain(handler))
    await asyncio.gather(*(spill.append(record) for record in second))

    assert await drain_task + await spill.depth() == 40


async def test_two_drains_at_once_do_not_insert_anything_twice(
    spill: SpillFile, make_records
) -> None:
    # A duplicated audit row is a quieter failure than a lost one and just as
    # wrong: the trail is supposed to say a request happened once.
    for record in make_records(6):
        await spill.append(record)

    drained: list[AuditRecord] = []

    async def slow_handler(batch) -> None:
        await asyncio.sleep(0.01)
        drained.extend(batch)

    counts = await asyncio.gather(spill.drain(slow_handler), spill.drain(slow_handler))

    assert sum(counts) == 6
    assert len(drained) == 6


async def test_the_file_is_json_lines(spill: SpillFile, make_records) -> None:
    # A human debugging an outage reads this file with `tail` and `jq`; keeping
    # it one record per line is part of the contract.
    for record in make_records(3):
        await spill.append(record)

    lines = spill.path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert all(line.startswith("{") and line.endswith("}") for line in lines)
