"""Ingestion — Technical Design §3.1, plus the two things §3.1 does not say.

The flow as written: normalise, hash, short-circuit on an identical hash, chunk,
carry forward the chunks whose hashes already exist, embed the remainder, write
Postgres in one transaction as `pending`, upsert Chroma, mark `indexed`.

**Carrying a chunk forward copies its vector, not its id.** The obvious reading
of AD-008 is that a reused chunk keeps the `chroma_id` its matching row already
has, so two chunk rows point at one vector. That breaks two things at once: the
vector's metadata names the *other* document, so a search result attributes the
chunk to the wrong one; and the delete path purges by `document_id` filter, so
deleting either document either strands a live chunk or leaves a vector behind.
Each chunk row therefore gets its own id, and reuse means fetching the stored
vector and writing it under the new id. The embedding call is still skipped,
which is the part that costs time and money. **AD-019**

**An ingest whose `source` matches a live document supersedes it.** §3.1 exits
on an identical hash and otherwise inserts; nothing said what becomes of the
previous version, so re-ingesting an edited file left both in the collection and
search answered with both. Vectors are fetched before anything is purged, so a
supersede still carries forward every unchanged chunk of the document it
replaces. **AD-020**

Ordering is what makes a crash survivable. Postgres is written first and marked
`pending`; Chroma is written second; the status flips last. The superseded
document is soft-deleted only after the new vectors are in, and its rows are
hard-deleted only after its vectors are out. Every gap leaves a marker the
startup reconciliation pass can act on, and every Chroma write is keyed by an id
derived from content, so replaying one overwrites rather than duplicates.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from uuid import UUID, uuid4

from ai_embeddings import EmbeddingVector
from kb_api.chunking import chunk_document, content_hash, normalise, to_new_chunk
from kb_api.domain import (
    Chunk,
    ChunkStore,
    DocumentFilter,
    DocumentStatus,
    DocumentStore,
    IpAddress,
    NewChunk,
    NewDocument,
    TelemetryPort,
    VectorRecord,
    VectorStore,
    chunk_metadata,
)
from kb_api.services.embedding import embed_in_batches
from kb_api.services.providers import ProviderRegistry, ResolvedProvider
from platform_core import ValidationError, get_logger, get_request_id
from platform_db import SessionSource

__all__ = ["IngestOutcome", "IngestRequest", "IngestResult", "IngestionService"]

_logger = get_logger("kb.ingestion")


class IngestOutcome(StrEnum):
    """PRD §6.3's `status`. `failed` is never returned — it is raised."""

    SUCCESS = "success"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class IngestRequest:
    """One document to ingest, with the provenance AD-014 wants recorded."""

    title: str
    content: str
    type: str
    source: str | None = None
    tags: tuple[str, ...] = ()
    provider: str | None = None
    ingested_by_key_id: str | None = None
    ingested_from_ip: IpAddress | None = None


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What §6.3 answers with. `embedding_calls` is not in the PRD response —
    it is what the AD-008 exit criteria are asserted on, and what tier-2
    telemetry records, so the flow reports it rather than a test inferring it."""

    document_id: UUID
    collection: str
    outcome: IngestOutcome
    chunks_created: int = 0
    chunks_reused: int = 0
    total_tokens: int = 0
    embedding_calls: int = 0
    superseded: tuple[UUID, ...] = field(default_factory=tuple)


class IngestionService:
    def __init__(
        self,
        *,
        sessions: SessionSource,
        documents: DocumentStore,
        chunks: ChunkStore,
        vectors: VectorStore,
        providers: ProviderRegistry,
        telemetry: TelemetryPort | None = None,
    ) -> None:
        self._sessions = sessions
        self._documents = documents
        self._chunks = chunks
        self._vectors = vectors
        self._providers = providers
        self._telemetry = telemetry

    async def ingest(self, request: IngestRequest) -> IngestResult:
        started = perf_counter()
        target = self._providers.resolve(request.provider)
        content = normalise(request.content)
        if not content.strip():
            raise ValidationError("Document content is empty.")

        digest = content_hash(content)
        async with self._sessions.session() as session:
            existing = await self._documents.find_by_content_hash(
                session, digest, target.collection
            )
        if existing is not None:
            # §3.1 step 2. Zero chunks, zero tokens, zero API calls — and no
            # supersede, since the live document already holds this content.
            _logger.info("ingest_unchanged", document_id=str(existing.id))
            self._emit_ingest(
                collection=existing.collection,
                outcome=IngestOutcome.UNCHANGED.value,
                document_id=existing.id,
                content_bytes=len(content.encode()),
                started=started,
            )
            return IngestResult(
                document_id=existing.id,
                collection=existing.collection,
                outcome=IngestOutcome.UNCHANGED,
            )

        document_id = uuid4()
        new_chunks = tuple(
            to_new_chunk(draft, document_id=document_id, model_id=target.provider.model.model_id)
            for draft in chunk_document(content)
        )
        if not new_chunks:
            raise ValidationError("Document content produced no chunks.")

        # Before Postgres holds a row claiming this collection exists. A Chroma
        # outage should fail the request outright rather than leave a `pending`
        # document pointing at a collection that was never created.
        await self._vectors.ensure_collection(target.collection)

        async with self._sessions.session() as session:
            reusable = await self._chunks.find_reusable(
                session, [chunk.text_hash for chunk in new_chunks], target.collection
            )
            superseded = (
                await self._documents.ids_matching(
                    session,
                    DocumentFilter(source=request.source, collection=target.collection),
                )
                if request.source is not None
                else ()
            )

        carried = await self._carry_forward(target.collection, reusable)
        embedded, tokens, calls = await self._embed(target, new_chunks, carried)
        vectors = {**carried, **embedded}

        records = tuple(
            VectorRecord(
                id=chunk.chroma_id,
                vector=vectors[chunk.text_hash],
                chunk_text=chunk.text,
                metadata=chunk_metadata(
                    document_id=document_id,
                    title=request.title,
                    source=request.source,
                    document_type=request.type,
                    tags=request.tags,
                    ordinal=chunk.ordinal,
                ),
            )
            for chunk in new_chunks
        )

        # §3.1 step 6 — document and chunks in one transaction, as `pending`.
        async with self._sessions.session() as session:
            await self._documents.add(
                session,
                NewDocument(
                    id=document_id,
                    title=request.title,
                    content=content,
                    content_hash=digest,
                    type=request.type,
                    collection=target.collection,
                    source=request.source,
                    tags=request.tags,
                    status=DocumentStatus.PENDING,
                    chunk_count=len(new_chunks),
                    ingested_by_key_id=request.ingested_by_key_id,
                    ingested_from_ip=request.ingested_from_ip,
                ),
            )
            await self._chunks.add_all(session, new_chunks)

        await self._vectors.upsert(target.collection, records)

        async with self._sessions.session() as session:
            await self._documents.set_status(
                session, document_id, DocumentStatus.INDEXED, chunk_count=len(new_chunks)
            )
            for stale in superseded:
                # Soft first. From here the old document is unsearchable through
                # Postgres, and its surviving chunk rows are the marker saying
                # its vectors still have to go.
                await self._documents.soft_delete(session, stale)

        for stale in superseded:
            await self._purge(target.collection, stale)

        reused = sum(1 for chunk in new_chunks if chunk.text_hash in carried)
        self._emit_ingest(
            collection=target.collection,
            outcome=IngestOutcome.SUCCESS.value,
            document_id=document_id,
            chunks_created=len(new_chunks) - reused,
            chunks_reused=reused,
            content_bytes=len(content.encode()),
            started=started,
        )
        _logger.info(
            "ingest_complete",
            document_id=str(document_id),
            collection=target.collection,
            chunks_created=len(new_chunks) - reused,
            chunks_reused=reused,
            embedding_calls=calls,
            tokens=tokens,
            superseded=len(superseded),
        )
        return IngestResult(
            document_id=document_id,
            collection=target.collection,
            outcome=IngestOutcome.SUCCESS,
            chunks_created=len(new_chunks) - reused,
            chunks_reused=reused,
            total_tokens=tokens,
            embedding_calls=calls,
            superseded=superseded,
        )

    async def _carry_forward(
        self, collection: str, reusable: dict[str, Chunk]
    ) -> dict[str, EmbeddingVector]:
        """Text hash → the vector already stored for it (AD-019).

        A hash whose vector is missing from Chroma is simply absent from the
        result and gets embedded like any other. That is drift between the two
        stores — AD-003 says Postgres wins and the index is rebuildable — so the
        right response is to rebuild the missing piece, not to fail the request.
        """
        if not reusable:
            return {}
        by_chroma_id = {chunk.chroma_id: text_hash for text_hash, chunk in reusable.items()}
        found = await self._vectors.fetch_vectors(collection, list(by_chroma_id))
        if len(found) != len(by_chroma_id):
            _logger.warning(
                "chunk_vectors_missing",
                collection=collection,
                expected=len(by_chroma_id),
                found=len(found),
            )
        return {by_chroma_id[chroma_id]: vector for chroma_id, vector in found.items()}

    async def _embed(
        self,
        target: ResolvedProvider,
        new_chunks: tuple[NewChunk, ...],
        carried: dict[str, EmbeddingVector],
    ) -> tuple[dict[str, EmbeddingVector], int, int]:
        """Embed what could not be carried forward. §3.1 step 5.

        Deduplicated by hash before batching: a document that repeats a passage
        chunks to two rows with one hash, and paying for the same vector twice
        is what AD-008 exists to stop — within a document as much as across one.
        """
        pending: dict[str, str] = {}
        for chunk in new_chunks:
            if chunk.text_hash not in carried:
                pending.setdefault(chunk.text_hash, chunk.text)
        if not pending:
            return {}, 0, 0

        result = await embed_in_batches(target.provider, list(pending.values()))
        if self._telemetry is not None and (request_id := _request_uuid()) is not None:
            model = target.provider.model
            self._telemetry.tokens_used(
                request_id=request_id,
                provider=target.name,
                model=model.model_id,
                operation="ingest",
                input_tokens=result.tokens,
                token_source=result.token_source,
                api_calls=result.calls,
            )
        embedded = dict(zip(pending, result.vectors, strict=True))
        return embedded, result.tokens, result.calls

    def _emit_ingest(
        self,
        *,
        collection: str,
        outcome: str,
        document_id: UUID,
        started: float,
        chunks_created: int = 0,
        chunks_reused: int = 0,
        content_bytes: int | None = None,
    ) -> None:
        """Tier 2, so it never raises and never blocks (AD-013).

        Skipped outside a request — the CLI and the reconciliation pass both
        call this flow with no correlation id, and a telemetry row that invented
        one would join to nothing.
        """
        if self._telemetry is None or (request_id := _request_uuid()) is None:
            return
        self._telemetry.ingest_completed(
            request_id=request_id,
            collection=collection,
            outcome=outcome,
            document_id=document_id,
            chunks_created=chunks_created,
            chunks_reused=chunks_reused,
            content_bytes=content_bytes,
            duration_ms=round((perf_counter() - started) * 1000),
        )

    async def _purge(self, collection: str, document_id: UUID) -> None:
        """Vectors out, then rows out. Never the other way round.

        The chunk rows are what says the purge is still owed; deleting them
        first would erase the evidence and strand the vectors in a collection
        nothing points at.
        """
        await self._vectors.delete_document(collection, document_id)
        async with self._sessions.session() as session:
            await self._chunks.delete_for_document(session, document_id)
            await self._documents.hard_delete(session, document_id)


def _request_uuid() -> UUID | None:
    """The correlation id, when there is one.

    `RequestContextMiddleware` binds it per request; a CLI or startup caller has
    none, and the tier-2 tables key on it.
    """
    request_id = get_request_id()
    return UUID(request_id) if request_id else None
