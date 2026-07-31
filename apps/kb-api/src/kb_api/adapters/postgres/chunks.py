"""Chunk persistence — batch writes and the AD-008 reuse lookup.

Chunks are only ever written a document at a time, so everything here is
plural. A per-chunk insert loop over a 40-chunk document is 40 round trips
inside the transaction that the ingest request is waiting on.
"""

from collections.abc import Sequence
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, Row, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kb_api.adapters.postgres.tables import chunks, documents
from kb_api.domain import Chunk, NewChunk

__all__ = ["ChunkRepository"]


class ChunkRepository:
    """Batch operations over `kb.chunks`."""

    async def add_all(self, session: AsyncSession, new_chunks: Sequence[NewChunk]) -> int:
        """Insert a document's chunks in one statement. Returns how many.

        An empty sequence is a no-op rather than an error: a document whose
        content chunks to nothing is a chunker question, and answering it with
        a database exception would put it in the wrong layer.
        """
        if not new_chunks:
            return 0
        await session.execute(
            chunks.insert(),
            [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                    "text_hash": chunk.text_hash,
                    "token_count": chunk.token_count,
                    "chroma_id": chunk.chroma_id,
                }
                for chunk in new_chunks
            ],
        )
        return len(new_chunks)

    async def for_document(self, session: AsyncSession, document_id: UUID) -> tuple[Chunk, ...]:
        """Every chunk of one document, in ordinal order."""
        result = await session.execute(
            select(chunks).where(chunks.c.document_id == document_id).order_by(chunks.c.ordinal)
        )
        return tuple(_to_chunk(row) for row in result.all())

    async def find_reusable(
        self, session: AsyncSession, text_hashes: Sequence[str], collection: str
    ) -> dict[str, Chunk]:
        """The AD-008 carry-forward lookup: which of these chunks already exist?

        Keyed by `text_hash` because that is what the caller holds — it has just
        chunked and hashed a document and wants to know which pieces it can skip
        embedding.

        Scoped to the collection through a join on the parent document. The hash
        already covers the model (`sha256(text + model_id)`), but the *vector*
        being carried forward lives in one collection, and reusing a
        `chroma_id` from another one would point at a vector that is not there.
        Live documents only: a chunk of a deleted document has had its vector
        purged from Chroma, so it is a hash match with nothing behind it.

        Duplicate hashes within a collection — the same paragraph in two
        documents — collapse to one entry. Any of them is as good as any other:
        the caller wants the text and token count, and re-embedding identical
        text is exactly what this avoids.
        """
        if not text_hashes:
            return {}
        result = await session.execute(
            select(chunks)
            .join(documents, documents.c.id == chunks.c.document_id)
            .where(
                chunks.c.text_hash.in_(set(text_hashes)),
                documents.c.collection == collection,
                documents.c.deleted_at.is_(None),
            )
        )
        return {row.text_hash: _to_chunk(row) for row in result.all()}

    async def delete_for_document(self, session: AsyncSession, document_id: UUID) -> int:
        """Hard-delete a document's chunks. Returns how many rows went.

        Runs after the Chroma purge in the delete flow (Design §3.3). The
        document row's `deleted_at` is the marker that survives a crash between
        the two, so losing these rows early would leave nothing to retry from.
        """
        result = cast(
            "CursorResult[Any]",
            await session.execute(delete(chunks).where(chunks.c.document_id == document_id)),
        )
        return result.rowcount

    async def count_for_document(self, session: AsyncSession, document_id: UUID) -> int:
        result = await session.execute(
            select(func.count()).select_from(chunks).where(chunks.c.document_id == document_id)
        )
        return int(result.scalar_one())


def _to_chunk(row: Row[Any]) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        ordinal=row.ordinal,
        text=row.text,
        text_hash=row.text_hash,
        token_count=row.token_count,
        chroma_id=row.chroma_id,
        created_at=row.created_at,
    )
