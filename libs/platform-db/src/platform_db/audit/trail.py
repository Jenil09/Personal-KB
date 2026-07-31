"""Tier 1 — the security trail that is never dropped (AD-013).

One synchronous insert per request, before the response is returned. It costs
2-5 ms against a co-located Postgres, which is noise beside the 200-400 ms
embedding call that dominates the request (Design §3.2).

The two rules that shape everything here:

- **A record is never discarded.** Postgres unreachable means the record goes to
  the spill file and a reconciler drains it on recovery.
- **A request never fails because the trail is unavailable.** Fail-closed is
  right for a bank and wrong for a personal KB that a Postgres restart would
  then brick.

`record()` therefore raises nothing at all. That is deliberate, and it is why
the spill failure path logs at `critical` with the record inlined: if both
Postgres and the disk are gone, the JSON log is the last durable surface left
and the record has to be recoverable from it.
"""

import asyncio
from collections.abc import Sequence
from contextlib import suppress

from sqlalchemy import Table, insert
from sqlalchemy.exc import SQLAlchemyError

from platform_core import get_logger
from platform_db.audit.records import AuditRecord
from platform_db.audit.spill import SpillFile
from platform_db.audit.tables import request_logs
from platform_db.engine import SessionSource

__all__ = ["AuditTrail"]

_logger = get_logger("platform.audit.trail")


class AuditTrail:
    """Writes tier-1 records, and drains whatever the outage left behind."""

    def __init__(
        self,
        database: SessionSource,
        spill: SpillFile,
        *,
        reconcile_interval_seconds: float = 30.0,
        table: Table = request_logs,
    ) -> None:
        self._database = database
        self._spill = spill
        self._interval = reconcile_interval_seconds
        self._table = table
        self._reconciler: asyncio.Task[None] | None = None

    async def record(self, record: AuditRecord) -> None:
        """Persist one record. Never raises, never drops."""
        try:
            await self._insert([record])
        except (SQLAlchemyError, OSError) as exc:
            _logger.warning(
                "audit_write_spilled",
                request_id=str(record.request_id),
                error=str(exc),
            )
            await self._spill_record(record)

    async def reconcile(self) -> int:
        """Drain the spill into Postgres. Returns the number of records moved."""
        drained = await self._spill.drain(self._insert)
        if drained:
            _logger.info("audit_spill_drained", records=drained)
        return drained

    async def spill_depth(self) -> int:
        """Records still awaiting reconciliation. Non-zero means degraded health."""
        return await self._spill.depth()

    async def start(self) -> None:
        """Begin the periodic drain. Called from the lifespan handler.

        Runs once immediately, so a service restarted after an outage does not
        sit on a full spill file for a whole interval before noticing.
        """
        if self._reconciler is not None:
            return
        self._reconciler = asyncio.create_task(self._reconcile_forever())

    async def stop(self) -> None:
        """Stop the reconciler. Anything left in the spill stays on disk."""
        if self._reconciler is None:
            return
        self._reconciler.cancel()
        with suppress(asyncio.CancelledError):
            await self._reconciler
        self._reconciler = None

    async def _reconcile_forever(self) -> None:
        while True:
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Postgres still down, or down again. The records are on disk;
                # the only correct response is to come back and try later.
                _logger.warning("audit_reconcile_failed", exc_info=exc)
            await asyncio.sleep(self._interval)

    async def _insert(self, records: Sequence[AuditRecord]) -> None:
        if not records:
            return
        async with self._database.session() as session:
            await session.execute(insert(self._table), [record.to_row() for record in records])

    async def _spill_record(self, record: AuditRecord) -> None:
        try:
            await self._spill.append(record)
        except OSError as exc:
            _logger.critical(
                "audit_record_unwritable",
                error=str(exc),
                record=record.model_dump(mode="json"),
            )
