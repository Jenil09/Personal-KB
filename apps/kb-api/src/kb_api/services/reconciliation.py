"""Startup reconciliation — closing the windows the two-store write leaves.

AD-003 makes Postgres the system of record and Chroma a derived index, which is
what makes this possible at all: every state a crash can leave is one Postgres
still describes, and the index can be rebuilt from it.

Two windows, two markers.

**`status = 'pending'`** — the document and its chunks are committed and the
vectors may not be. Design §3.1's own note. Fixed by asking Chroma which of the
document's `chroma_id`s it already holds and re-embedding only the rest, so the
common case (the crash landed after the upsert) costs one Chroma read and no API
calls at all.

**Soft-deleted with chunk rows surviving** — the vectors are still in the index
for a document Postgres considers gone. §3.3 names the soft-delete flag as the
crash marker for the delete path; the surviving chunk rows are what says the
purge never ran, because the hard delete is the last step. Left alone this is
the worse of the two: search reads Chroma alone (AD-004), so it answers with a
document nothing else in the system admits exists.

This runs once at startup, in the lifespan handler, before the app serves. It is
deliberately not a background loop: at 1-4 ingests a month a repair that waits
for the next restart is not a real exposure, and a loop would need leader
election the moment a second worker appeared.
"""

from dataclasses import dataclass
from uuid import UUID

from kb_api.domain import (
    Chunk,
    ChunkStore,
    Document,
    DocumentStatus,
    DocumentStore,
    VectorRecord,
    VectorStore,
    chunk_metadata,
)
from kb_api.services.embedding import embed_in_batches
from kb_api.services.providers import ProviderRegistry, ResolvedProvider
from platform_core import ConfigurationError, PlatformError, get_logger
from platform_db import SessionSource

__all__ = ["ReconciliationReport", "ReconciliationService"]

_logger = get_logger("kb.reconciliation")


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """What the pass did. Logged at startup and surfaced in Phase 8's stats."""

    reindexed: int = 0
    purged: int = 0
    failed: int = 0

    @property
    def clean(self) -> bool:
        return not (self.reindexed or self.purged or self.failed)


class ReconciliationService:
    def __init__(
        self,
        *,
        sessions: SessionSource,
        documents: DocumentStore,
        chunks: ChunkStore,
        vectors: VectorStore,
        providers: ProviderRegistry,
        batch_limit: int = 100,
    ) -> None:
        self._sessions = sessions
        self._documents = documents
        self._chunks = chunks
        self._vectors = vectors
        self._providers = providers
        self._batch_limit = batch_limit

    async def run(self) -> ReconciliationReport:
        """Repair both windows. Never raises — startup must not hinge on it.

        A document that cannot be repaired is marked `failed` and logged. The
        alternative, refusing to start, takes a working service down over one
        bad row and leaves the operator no endpoint to ask about it from.
        """
        async with self._sessions.session() as session:
            pending = await self._documents.find_pending(session, limit=self._batch_limit)
            purgeable = await self._documents.find_purgeable(session, limit=self._batch_limit)

        reindexed = failed = 0
        for document in pending:
            if await self._reindex(document):
                reindexed += 1
            else:
                failed += 1

        purged = 0
        for document in purgeable:
            if await self._purge(document):
                purged += 1
            else:
                failed += 1

        report = ReconciliationReport(reindexed=reindexed, purged=purged, failed=failed)
        if report.clean:
            _logger.info("reconciliation_clean")
        else:
            _logger.warning(
                "reconciliation_repaired",
                reindexed=report.reindexed,
                purged=report.purged,
                failed=report.failed,
            )
        return report

    async def _reindex(self, document: Document) -> bool:
        try:
            async with self._sessions.session() as session:
                rows = await self._chunks.for_document(session, document.id)
            if not rows:
                # Committed the document but not its chunks, or a chunker that
                # produced none. Nothing to index and nothing to recover from —
                # the content is still in Postgres, so a re-ingest fixes it.
                await self._mark(document.id, DocumentStatus.FAILED)
                _logger.warning("reconciliation_no_chunks", document_id=str(document.id))
                return False

            await self._vectors.ensure_collection(document.collection)
            missing = await self._missing_vectors(document.collection, rows)
            if missing:
                await self._vectors.upsert(
                    document.collection, await self._rebuild(document, missing)
                )
            await self._mark(document.id, DocumentStatus.INDEXED, chunk_count=len(rows))
            _logger.info(
                "reconciliation_reindexed",
                document_id=str(document.id),
                chunks=len(rows),
                re_embedded=len(missing),
            )
        except PlatformError as exc:
            _logger.error(
                "reconciliation_reindex_failed", document_id=str(document.id), exc_info=exc
            )
            return False
        return True

    async def _purge(self, document: Document) -> bool:
        try:
            await self._vectors.delete_document(document.collection, document.id)
            async with self._sessions.session() as session:
                await self._chunks.delete_for_document(session, document.id)
                await self._documents.hard_delete(session, document.id)
            _logger.info("reconciliation_purged", document_id=str(document.id))
        except PlatformError as exc:
            _logger.error("reconciliation_purge_failed", document_id=str(document.id), exc_info=exc)
            return False
        return True

    async def _missing_vectors(self, collection: str, rows: tuple[Chunk, ...]) -> tuple[Chunk, ...]:
        """The chunks Chroma does not already hold.

        A crash after the upsert — the likely one, since the upsert is what
        precedes the status flip — finds everything present and re-embeds
        nothing.
        """
        present = await self._vectors.fetch_vectors(collection, [chunk.chroma_id for chunk in rows])
        return tuple(chunk for chunk in rows if chunk.chroma_id not in present)

    async def _rebuild(
        self, document: Document, missing: tuple[Chunk, ...]
    ) -> tuple[VectorRecord, ...]:
        """Re-embed the absent chunks from the text Postgres kept (AD-003).

        The provider comes from the document's own collection rather than from
        whatever is configured as default, so a document written under a
        provider that has since been unconfigured is reported rather than
        silently re-embedded into a different space (AD-006).
        """
        target = self._provider_for(document.collection)
        result = await embed_in_batches(target.provider, [chunk.text for chunk in missing])
        return tuple(
            VectorRecord(
                id=chunk.chroma_id,
                vector=vector,
                chunk_text=chunk.text,
                metadata=chunk_metadata(
                    document_id=document.id,
                    title=document.title,
                    source=document.source,
                    document_type=document.type,
                    tags=document.tags,
                    ordinal=chunk.ordinal,
                ),
            )
            for chunk, vector in zip(missing, result.vectors, strict=True)
        )

    def _provider_for(self, collection: str) -> ResolvedProvider:
        for name in self._providers.names:
            candidate = self._providers.resolve(name)
            if candidate.collection == collection:
                return candidate
        raise ConfigurationError(
            f"no configured provider writes to collection {collection}",
            context={"collection": collection},
        )

    async def _mark(
        self, document_id: UUID, status: DocumentStatus, *, chunk_count: int | None = None
    ) -> None:
        async with self._sessions.session() as session:
            await self._documents.set_status(session, document_id, status, chunk_count=chunk_count)
