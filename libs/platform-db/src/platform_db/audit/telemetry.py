"""Tier 2 — best-effort telemetry, off the request path (AD-009, retained by AD-013).

Token counts, per-stage timings, cache hit rates, embedding call counts. High
volume, low individual value: losing one is uninteresting, and paying a
synchronous insert for each would multiply AD-013's one-insert-per-request into
several.

A bounded queue with a batching consumer. When the queue is full, records are
**dropped and counted** — that is the contract, not a failure mode. The drop
counter is exposed so the loss is visible rather than silent; an unbounded queue
would trade a counted drop for an unbounded memory ceiling on a 4 GB box.

The sink names no tables. Tier-2 shapes are per-service, so an event carries the
`Table` it belongs in and the consumer groups a batch by table before inserting.
"""

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Table, insert
from sqlalchemy.exc import SQLAlchemyError

from platform_core import get_logger
from platform_db.engine import SessionSource
from platform_db.settings import AuditSettings

__all__ = ["TelemetryEvent", "TelemetrySink"]

_logger = get_logger("platform.audit.telemetry")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One row destined for `table`.

    `Any` in the values: a telemetry row is whatever its own table declares, and
    the sink deliberately knows nothing about that shape.
    """

    table: Table
    values: Mapping[str, Any] = field(default_factory=dict)


class TelemetrySink:
    """Bounded queue, batching consumer, explicit drop counter."""

    def __init__(self, database: SessionSource, settings: AuditSettings) -> None:
        self._database = database
        self._settings = settings
        self._queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue(
            maxsize=settings.telemetry_queue_size
        )
        self._consumer: asyncio.Task[None] | None = None
        self._dropped = 0
        self._written = 0

    @property
    def dropped(self) -> int:
        """Records lost to a full queue. Surfaced on `/health` and admin stats."""
        return self._dropped

    @property
    def written(self) -> int:
        return self._written

    @property
    def depth(self) -> int:
        """Queue depth, for the health endpoint's saturation signal."""
        return self._queue.qsize()

    def emit(self, event: TelemetryEvent) -> None:
        """Enqueue without blocking or awaiting.

        Synchronous on purpose: a caller emitting telemetry is in the middle of
        serving a request and must not be able to wait on the sink, however
        briefly. A full queue drops here rather than backing pressure up into
        the request path.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1

    async def start(self) -> None:
        if self._consumer is not None:
            return
        self._consumer = asyncio.create_task(self._consume_forever())

    async def flush(self) -> None:
        """Wait for the queue to empty, up to the configured drain timeout.

        Times out rather than hangs: tier 2 is best-effort, and a wedged
        Postgres must not keep the process from shutting down.
        """
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=self._settings.telemetry_drain_timeout_seconds,
            )
        except TimeoutError:
            abandoned = self._queue.qsize()
            self._dropped += abandoned
            _logger.warning("telemetry_flush_timeout", abandoned=abandoned)

    async def stop(self) -> None:
        """Flush what is queued, then stop the consumer."""
        if self._consumer is None:
            return
        await self.flush()
        self._consumer.cancel()
        with suppress(asyncio.CancelledError):
            await self._consumer
        self._consumer = None

    async def _consume_forever(self) -> None:
        while True:
            batch = await self._collect_batch()
            try:
                await self._write(batch)
                self._written += len(batch)
            except (SQLAlchemyError, OSError) as exc:
                # Tier 2 is allowed to lose these. Tier 1 is the trail that is
                # not, and it does not come through here.
                self._dropped += len(batch)
                _logger.warning("telemetry_batch_failed", records=len(batch), error=str(exc))
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _collect_batch(self) -> list[TelemetryEvent]:
        """Up to `batch_size` events, or whatever arrived within the interval.

        Waits indefinitely for the first event — an idle service should not be
        waking up to insert nothing — and only then opens the window.
        """
        batch = [await self._queue.get()]
        deadline = (
            asyncio.get_running_loop().time() + self._settings.telemetry_flush_interval_seconds
        )
        while len(batch) < self._settings.telemetry_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return batch

    async def _write(self, batch: Sequence[TelemetryEvent]) -> None:
        grouped: dict[Table, list[Mapping[str, Any]]] = {}
        for event in batch:
            grouped.setdefault(event.table, []).append(event.values)
        async with self._database.session() as session:
            for table, rows in grouped.items():
                await session.execute(insert(table), [dict(row) for row in rows])
