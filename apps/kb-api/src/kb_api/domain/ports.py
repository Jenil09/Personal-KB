"""The ports the services bind to (Design §1, layering rule).

Phase 4 deliberately shipped the repositories without these: a `Protocol`
written before its caller describes a guess rather than a requirement. The
ingestion service is that caller, so the seams it actually uses get named here
and nothing else does — every method below is one the service calls.

They are `Protocol`s rather than ABCs so the concrete repositories in
`adapters/postgres` satisfy them without importing anything from `domain`, which
keeps the dependency arrow pointing the way the layering rule requires.

`VectorStore` is the one that matters. AD-010 keeps the Chroma adapter inside
this service rather than in `libs/`, on the grounds that there is no second
consumer yet; the port is what makes that a cheap decision to revisit instead of
a commitment. Nothing above `adapters/chroma` imports `chromadb`.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ai_embeddings import EmbeddingVector
from kb_api.domain.documents import (
    Chunk,
    Document,
    DocumentFilter,
    DocumentPage,
    DocumentStatus,
    NewChunk,
    NewDocument,
)
from kb_api.domain.vectors import VectorMatch, VectorRecord

__all__ = ["ChunkStore", "DocumentStore", "TelemetryPort", "VectorStore"]


class VectorStore(Protocol):
    """The index. Rebuildable from Postgres by definition (AD-003)."""

    async def ensure_collection(self, name: str) -> None:
        """Create the collection if it is absent, cosine-spaced (Design §2.3).

        Idempotent, and called on the ingest path rather than at startup: the
        collection a request targets depends on the provider it names, and
        pre-creating every collection a service *could* address would populate
        the store with empty indexes nobody asked for.
        """

    async def collection_exists(self, name: str) -> bool:
        """Whether the collection has been created. Search's `409` (Design §3.2)."""

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Write vectors. Keyed by id, so replaying an interrupted ingest is safe."""

    async def fetch_vectors(
        self, collection: str, ids: Sequence[str]
    ) -> Mapping[str, EmbeddingVector]:
        """The vectors already stored under these ids; absent ids are omitted.

        This is what makes AD-008's carry-forward real. A chunk whose text hash
        already exists has a vector in the store, and copying it costs a local
        round trip instead of an embedding call.
        """

    async def delete_document(self, collection: str, document_id: UUID) -> int:
        """Purge one document's vectors by metadata filter. Returns how many went."""

    async def query(
        self,
        collection: str,
        vector: EmbeddingVector,
        *,
        top_k: int,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorMatch, ...]:
        """Nearest neighbours, nearest first, carrying Chroma's raw distance."""

    async def count(self, collection: str) -> int:
        """How many vectors the collection holds. Zero for a collection that exists."""

    async def heartbeat(self) -> bool:
        """False rather than raising — this backs a health check."""


class DocumentStore(Protocol):
    """The `kb.documents` operations the ingestion and reconciliation flows use."""

    async def add(self, session: AsyncSession, document: NewDocument) -> Document: ...

    async def get(
        self, session: AsyncSession, document_id: UUID, *, include_deleted: bool = False
    ) -> Document | None: ...

    async def find_by_content_hash(
        self, session: AsyncSession, content_hash: str, collection: str
    ) -> Document | None: ...

    async def ids_matching(
        self, session: AsyncSession, filters: DocumentFilter, *, limit: int | None = None
    ) -> tuple[UUID, ...]: ...

    async def list(
        self,
        session: AsyncSession,
        filters: DocumentFilter | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentPage: ...

    async def find_pending(
        self, session: AsyncSession, *, collection: str | None = None, limit: int = 100
    ) -> tuple[Document, ...]: ...

    async def find_purgeable(
        self, session: AsyncSession, *, collection: str | None = None, limit: int = 100
    ) -> tuple[Document, ...]: ...

    async def set_status(
        self,
        session: AsyncSession,
        document_id: UUID,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
    ) -> Document | None: ...

    async def soft_delete(self, session: AsyncSession, document_id: UUID) -> bool: ...

    async def hard_delete(self, session: AsyncSession, document_id: UUID) -> bool: ...


class ChunkStore(Protocol):
    """The `kb.chunks` operations the same flows use."""

    async def add_all(self, session: AsyncSession, new_chunks: Sequence[NewChunk]) -> int: ...

    async def for_document(self, session: AsyncSession, document_id: UUID) -> tuple[Chunk, ...]: ...

    async def find_reusable(
        self, session: AsyncSession, text_hashes: Sequence[str], collection: str
    ) -> dict[str, Chunk]: ...

    async def delete_for_document(self, session: AsyncSession, document_id: UUID) -> int: ...


class TelemetryPort(Protocol):
    """Tier-2 emission, as the flows see it (AD-013).

    Three verbs, no tables. Which table an event lands in is the adapter's
    knowledge (`adapters/postgres/telemetry.py`), and a service that knew it
    would be importing the schema it is supposed to be insulated from.

    Every method returns `None` and none of them are `async`. Tier 2 is
    best-effort and off the request path by construction: a flow emitting an
    event is in the middle of serving a request, so it must not be able to wait
    on the sink, and it must not be able to fail because of one either.
    """

    def tokens_used(
        self,
        *,
        request_id: UUID,
        provider: str,
        model: str,
        operation: str,
        input_tokens: int,
        token_source: str,
        api_calls: int = 1,
        billable_characters: int | None = None,
    ) -> None: ...

    def ingest_completed(
        self,
        *,
        request_id: UUID,
        collection: str,
        outcome: str,
        document_id: UUID | None = None,
        chunks_created: int = 0,
        chunks_reused: int = 0,
        content_bytes: int | None = None,
        duration_ms: int | None = None,
        stage_timings_ms: Mapping[str, int] | None = None,
    ) -> None: ...

    def error_occurred(
        self,
        *,
        request_id: UUID,
        error_code: str,
        exception_type: str,
        message: str,
        stack: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None: ...
