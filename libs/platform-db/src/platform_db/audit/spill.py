"""The fsync'd JSONL spill file behind AD-013's never-dropped guarantee.

When Postgres is unreachable a tier-1 record is appended here instead, and a
reconciler drains it on recovery. Two properties make that a guarantee rather
than a hope:

**Every append is fsync'd before it is acknowledged.** Not just flushed —
a flushed write sits in the page cache and a power loss takes it with it. The
cost is one fsync per audit-writing request, and it is only paid while Postgres
is down.

**Draining rotates rather than reads-then-truncates.** The file is renamed to a
sibling `.draining` first, which is atomic within a directory, so a crash
between the read and the delete leaves the records in a file the next drain
picks up. Read-then-truncate loses everything written since the read; a truncate
that lands before the insert commits loses the batch outright.

File I/O runs in a worker thread. `fsync` blocks for milliseconds and the event
loop is also serving requests.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from platform_core import get_logger
from platform_db.audit.records import AuditRecord

__all__ = ["SpillFile"]

_logger = get_logger("platform.audit.spill")

_DRAINING_SUFFIX = ".draining"


class SpillFile:
    """Append-only JSONL, one record per line, with rotate-on-drain."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._draining_path = path.with_name(path.name + _DRAINING_SUFFIX)
        # Serialises appends against a rotation. Without it a drain can rename
        # the file out from under a write that has already opened it, and that
        # record lands in a file nothing will ever look at again.
        self._lock = asyncio.Lock()
        # Serialises drains against each other, and only against each other.
        # Two overlapping drains both find the same `.draining` file and both
        # hand it to the handler, which inserts every record twice. Appends are
        # deliberately not held up by this: they happen on the request path.
        self._drain_lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def append(self, record: AuditRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append_sync, record)

    async def depth(self) -> int:
        """Records awaiting reconciliation, including a batch mid-drain.

        `/health` reports degraded while this is non-zero (Design §5).
        """
        async with self._lock:
            return await asyncio.to_thread(self._depth_sync)

    async def drain(self, handler: Callable[[Sequence[AuditRecord]], Awaitable[None]]) -> int:
        """Hand every spilled record to `handler`, then delete them.

        Returns the number of records drained. If `handler` raises, the file is
        left in place and the exception propagates — the caller's job is to try
        again later, not to decide the records are expendable.

        Records that arrive during the drain go to a fresh spill file and are
        picked up by the next call.
        """
        async with self._drain_lock:
            async with self._lock:
                if not await asyncio.to_thread(self._rotate_sync):
                    return 0

            records = await asyncio.to_thread(self._read_draining_sync)
            if records:
                await handler(records)
            await asyncio.to_thread(self._draining_path.unlink, True)
            return len(records)

    # --- thread-side ------------------------------------------------------

    def _append_sync(self, record: AuditRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _depth_sync(self) -> int:
        return sum(self._count_lines(path) for path in (self._path, self._draining_path))

    def _rotate_sync(self) -> bool:
        """Move the spill aside for reading. False when there is nothing to drain.

        A `.draining` file already present is a crashed drain, and takes
        priority: the live spill is left alone and picked up on the next pass,
        so records are never reordered past a batch that is already in hand.
        """
        if self._draining_path.exists():
            return True
        if not self._path.exists():
            return False
        self._path.replace(self._draining_path)
        return True

    def _read_draining_sync(self) -> list[AuditRecord]:
        records: list[AuditRecord] = []
        with self._draining_path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not (stripped := line.strip()):
                    continue
                try:
                    records.append(AuditRecord.model_validate_json(stripped))
                except ValueError as exc:
                    # A torn final line is the one corruption fsync cannot rule
                    # out. Skipping it saves the rest of the file; failing the
                    # drain would strand every record behind it forever.
                    _logger.error(
                        "audit_spill_line_unreadable",
                        path=str(self._draining_path),
                        line_number=number,
                        error=str(exc),
                    )
        return records

    @staticmethod
    def _count_lines(path: Path) -> int:
        try:
            with path.open("rb") as handle:
                return sum(1 for line in handle if line.strip())
        except FileNotFoundError:
            return 0
