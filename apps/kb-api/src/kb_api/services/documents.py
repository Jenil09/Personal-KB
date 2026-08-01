"""Document management — PRD §6.4-6.6, and the delete half of Design §3.3.

Listing and reading are thin: the repository already returns a `DocumentPage`,
and the service exists to hold the layering rather than to add behaviour. Delete
is the one with a flow worth writing down.

**Delete is soft, then purge, then hard, in that order.** The soft flag goes in
first, so from that instant the document is unsearchable through Postgres and its
surviving chunk rows are the marker saying its vectors still have to go. The
vectors go next. The rows go last, because they are the evidence that the purge
is still owed — deleting them first would erase the marker and strand the vectors
in a collection nothing points at. Every gap in that sequence leaves a state the
startup reconciliation pass already knows how to finish
(`services/reconciliation.py`).

**A delete of a document that is not there is a success.** PRD §6.6 says `204`
whether or not it existed, which is what makes a retried delete safe — and a
retry is the expected case, since the client that timed out mid-delete has no
way to tell which side of the purge it stopped on. `soft_delete` returning False
distinguishes "already gone" from "never existed" without a second query, and
neither of them is an error.

**A document is deleted from the collection it is in, not from the one the
caller names.** Nothing in the delete request identifies a provider, and the
document itself knows which collection holds its vectors (AD-006). Reading it
first and using its own `collection` is what keeps a delete from silently
purging nothing when a second provider's corpus exists.
"""

from dataclasses import dataclass
from uuid import UUID

from kb_api.domain import (
    ChunkStore,
    Document,
    DocumentFilter,
    DocumentPage,
    DocumentStore,
    VectorStore,
)
from platform_core import NotFoundError, get_logger
from platform_db import SessionSource

__all__ = ["DeleteResult", "DocumentService"]

_logger = get_logger("kb.documents")


@dataclass(frozen=True, slots=True)
class DeleteResult:
    """What a delete did. The endpoint answers `204` either way (PRD §6.6)."""

    deleted: bool
    """False when the document was already gone — not an error, and not a 404."""

    vectors_purged: int = 0
    chunks_removed: int = 0


class DocumentService:
    def __init__(
        self,
        *,
        sessions: SessionSource,
        documents: DocumentStore,
        chunks: ChunkStore,
        vectors: VectorStore,
    ) -> None:
        self._sessions = sessions
        self._documents = documents
        self._chunks = chunks
        self._vectors = vectors

    async def list(
        self, filters: DocumentFilter | None = None, *, limit: int = 50, offset: int = 0
    ) -> DocumentPage:
        async with self._sessions.session() as session:
            return await self._documents.list(session, filters, limit=limit, offset=offset)

    async def get(self, document_id: UUID) -> Document:
        """One document, content included (PRD §6.5). `404` when it is not there."""
        async with self._sessions.session() as session:
            document = await self._documents.get(session, document_id)
        if document is None:
            raise NotFoundError(
                "No document with that id.", context={"document_id": str(document_id)}
            )
        return document

    async def delete(self, document_id: UUID) -> DeleteResult:
        """Remove a document from both stores. Idempotent (PRD §6.6, Design §3.3)."""
        async with self._sessions.session() as session:
            # Read before flagging: the document's own `collection` is the only
            # thing that says where its vectors live, and after the soft delete
            # a plain `get` would no longer return it.
            document = await self._documents.get(session, document_id, include_deleted=True)
            if document is None:
                return DeleteResult(deleted=False)
            newly_flagged = await self._documents.soft_delete(session, document_id)

        purged = await self._vectors.delete_document(document.collection, document_id)

        async with self._sessions.session() as session:
            removed = await self._chunks.delete_for_document(session, document_id)
            await self._documents.hard_delete(session, document_id)

        _logger.info(
            "document_deleted",
            document_id=str(document_id),
            collection=document.collection,
            vectors_purged=purged,
            chunks_removed=removed,
            # False means a previous attempt had already flagged it and stopped
            # somewhere before the rows went. This call finished that one.
            newly_flagged=newly_flagged,
        )
        return DeleteResult(deleted=True, vectors_purged=purged, chunks_removed=removed)
