"""Document persistence.

Every method takes the session rather than holding one. The session is the
request's transaction boundary and `Database.session()` owns it (Design §5), so
a repository that cached one would be deciding when the caller's work commits.
It also means the ingestion flow can write documents and chunks in the one
transaction §3.1 step 6 requires, by handing both repositories the same session.

Nothing here commits. Nothing here raises `HTTPException`. A missing document is
`None` and the service decides whether that is a `404`.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import (
    ColumnElement,
    CursorResult,
    Row,
    and_,
    delete,
    exists,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession

from kb_api.adapters.postgres.tables import chunks, documents
from kb_api.domain import (
    Document,
    DocumentFilter,
    DocumentPage,
    DocumentStatus,
    NewDocument,
)

__all__ = ["DocumentRepository"]

_ALIVE = documents.c.deleted_at.is_(None)


class DocumentRepository:
    """CRUD over `kb.documents`, plus the two lookups the flows depend on."""

    async def add(self, session: AsyncSession, document: NewDocument) -> Document:
        """Insert and return the row Postgres actually stored.

        `RETURNING *` rather than re-reading: `created_at`, `updated_at`, and
        the defaults are server-side, and a second SELECT would be a second
        chance to read something else.
        """
        statement = (
            documents.insert()
            .values(
                id=document.id,
                title=document.title,
                content=document.content,
                content_hash=document.content_hash,
                source=document.source,
                type=document.type,
                tags=list(document.tags),
                collection=document.collection,
                status=document.status.value,
                chunk_count=document.chunk_count,
                ingested_by_key_id=document.ingested_by_key_id,
                ingested_from_ip=_ip_text(document),
            )
            .returning(*documents.c)
        )
        result = await session.execute(statement)
        return _to_document(result.one())

    async def get(
        self, session: AsyncSession, document_id: UUID, *, include_deleted: bool = False
    ) -> Document | None:
        """One document by ID. Deleted ones are invisible unless asked for.

        `include_deleted` exists for the delete path, which has to tell "already
        gone" from "never existed" to stay idempotent (Design §3.3).
        """
        statement = select(documents).where(documents.c.id == document_id)
        if not include_deleted:
            statement = statement.where(_ALIVE)
        result = await session.execute(statement)
        row = result.one_or_none()
        return _to_document(row) if row is not None else None

    async def find_by_content_hash(
        self, session: AsyncSession, content_hash: str, collection: str
    ) -> Document | None:
        """AD-008's short-circuit: has this exact content already been indexed here?

        Scoped to the collection because the same content under a second
        provider is genuinely un-ingested — its vectors do not exist (AD-006).
        Backed by `uq_documents_content_hash_collection`.
        """
        result = await session.execute(
            select(documents).where(
                documents.c.content_hash == content_hash,
                documents.c.collection == collection,
                _ALIVE,
            )
        )
        row = result.one_or_none()
        return _to_document(row) if row is not None else None

    async def ids_matching(
        self, session: AsyncSession, filters: DocumentFilter, *, limit: int | None = None
    ) -> tuple[UUID, ...]:
        """The AD-005 tag lookup: document IDs for Chroma's `$in` clause.

        IDs only. The search response is built from the Chroma payload alone
        (AD-004), so pulling whole rows here would fetch document content that
        the request has no use for.

        `limit` guards the `$in` clause: AD-005 notes it gets unwieldy past a few
        thousand IDs. At personal-KB scale the cap is never reached, and if it
        ever is, the caller sees a truncated list rather than a query that times
        out somewhere further down.
        """
        statement = select(documents.c.id).where(_ALIVE, *_conditions(filters))
        if limit is not None:
            statement = statement.limit(limit)
        result = await session.execute(statement)
        return tuple(result.scalars())

    async def list(
        self,
        session: AsyncSession,
        filters: DocumentFilter | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentPage:
        """Paginated listing for `GET /v1/documents` (PRD §6.4).

        Ordered by `created_at DESC, id DESC`. The tiebreak matters: two
        documents ingested in the same bulk run can share a timestamp to the
        microsecond, and an unstable sort makes pagination drop and repeat rows
        across pages.
        """
        conditions = [_ALIVE, *_conditions(filters or DocumentFilter())]
        rows = await session.execute(
            select(documents)
            .where(*conditions)
            .order_by(documents.c.created_at.desc(), documents.c.id.desc())
            .limit(limit)
            .offset(offset)
        )
        total = await session.execute(
            select(func.count()).select_from(documents).where(*conditions)
        )
        return DocumentPage(
            documents=tuple(_to_document(row) for row in rows.all()),
            total=int(total.scalar_one()),
            limit=limit,
            offset=offset,
        )

    async def find_pending(
        self, session: AsyncSession, *, collection: str | None = None, limit: int = 100
    ) -> tuple[Document, ...]:
        """Documents Postgres has and Chroma may not — the reconciliation input.

        A crash between §3.1's steps 6 and 8 leaves rows here. Backed by the
        partial `ix_documents_status`, which indexes only the statuses that are
        not `indexed`, so this stays cheap on a table where almost everything is.
        """
        statement = select(documents).where(
            documents.c.status == DocumentStatus.PENDING.value, _ALIVE
        )
        if collection is not None:
            statement = statement.where(documents.c.collection == collection)
        result = await session.execute(statement.order_by(documents.c.created_at).limit(limit))
        return tuple(_to_document(row) for row in result.all())

    async def find_purgeable(
        self, session: AsyncSession, *, collection: str | None = None, limit: int = 100
    ) -> tuple[Document, ...]:
        """Soft-deleted documents whose chunk rows are still here.

        The other half of the reconciliation input. Design §3.3 makes the
        soft-delete flag the crash marker for the delete path, and a supersede is
        a delete with an insert in front of it — so a crash between the flag and
        the Chroma purge leaves a document Postgres considers gone whose vectors
        search still answers with. Surviving chunk rows are what says the purge
        never ran: the hard delete is the last step, so their absence means it
        completed.

        No index backs this and none should. It runs at startup over a table
        whose ceiling is ~100 documents, and an index on `deleted_at` would be
        maintained on every write to serve a query that runs once a deploy.
        """
        statement = (
            select(documents)
            .where(
                documents.c.deleted_at.is_not(None),
                exists().where(chunks.c.document_id == documents.c.id),
            )
            .order_by(documents.c.deleted_at)
            .limit(limit)
        )
        if collection is not None:
            statement = statement.where(documents.c.collection == collection)
        result = await session.execute(statement)
        return tuple(_to_document(row) for row in result.all())

    async def set_status(
        self,
        session: AsyncSession,
        document_id: UUID,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
    ) -> Document | None:
        """Mark a document `indexed` or `failed`, optionally with its chunk count."""
        values: dict[str, Any] = {"status": status.value, "updated_at": _now()}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        result = await session.execute(
            update(documents)
            .where(documents.c.id == document_id, _ALIVE)
            .values(**values)
            .returning(*documents.c)
        )
        row = result.one_or_none()
        return _to_document(row) if row is not None else None

    async def update_metadata(
        self,
        session: AsyncSession,
        document_id: UUID,
        *,
        title: str | None = None,
        source: str | None = None,
        document_type: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> Document | None:
        """Edit the fields that do not invalidate the vectors.

        Content is absent on purpose. Changing it changes `content_hash` and
        every chunk under it, which is an ingest, not an update.
        """
        values: dict[str, Any] = {"updated_at": _now()}
        if title is not None:
            values["title"] = title
        if source is not None:
            values["source"] = source
        if document_type is not None:
            values["type"] = document_type
        if tags is not None:
            values["tags"] = list(tags)
        result = await session.execute(
            update(documents)
            .where(documents.c.id == document_id, _ALIVE)
            .values(**values)
            .returning(*documents.c)
        )
        row = result.one_or_none()
        return _to_document(row) if row is not None else None

    async def soft_delete(self, session: AsyncSession, document_id: UUID) -> bool:
        """Flag the document deleted. True if this call is what deleted it.

        Set *before* the Chroma purge, so a crash in between leaves a document
        that is already unsearchable rather than one Postgres has forgotten and
        Chroma still answers with. False on an already-deleted document is what
        keeps `DELETE` idempotent (Design §3.3) without a second query.
        """
        # `CursorResult` rather than `Result`: `rowcount` is what distinguishes
        # "this call deleted it" from "it was already gone", and only the DML
        # result carries it.
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                update(documents)
                .where(documents.c.id == document_id, _ALIVE)
                .values(deleted_at=_now(), updated_at=_now())
            ),
        )
        return result.rowcount > 0

    async def hard_delete(self, session: AsyncSession, document_id: UUID) -> bool:
        """Remove the row outright, cascading to its chunks.

        Not the delete path — that one is soft. This is for the CLI and for
        tests that need a clean table.
        """
        result = cast(
            "CursorResult[Any]",
            await session.execute(delete(documents).where(documents.c.id == document_id)),
        )
        return result.rowcount > 0


def _conditions(filters: DocumentFilter) -> list[ColumnElement[bool]]:
    """Filter clauses shared by the tag lookup and the listing.

    The tag clause is `@>` against a one-element array, OR-ed or AND-ed, rather
    than one `@>` against the whole list. Both forms use the GIN index, but the
    single containment check `tags @> '["a","b"]'` can only express all-of;
    building it per tag is what makes any-of expressible at all.
    """
    conditions: list[ColumnElement[bool]] = []
    if filters.type is not None:
        conditions.append(documents.c.type == filters.type)
    if filters.source is not None:
        conditions.append(documents.c.source == filters.source)
    if filters.collection is not None:
        conditions.append(documents.c.collection == filters.collection)
    if filters.tags:
        # `.contains()` on a JSONB column is `@>` with the right-hand side bound
        # as a JSONB parameter, which is the form the jsonb_path_ops index can
        # answer. An untyped bind arrives as `unknown` and gets a sequential scan.
        contains = [documents.c.tags.contains([tag]) for tag in filters.tags]
        conditions.append(and_(*contains) if filters.match_all_tags else or_(*contains))
    return conditions


def _now() -> datetime:
    return datetime.now(UTC)


def _ip_text(document: NewDocument) -> str | None:
    # asyncpg wants a string for INET; `ipaddress` objects go in as text.
    return str(document.ingested_from_ip) if document.ingested_from_ip is not None else None


def _to_document(row: Row[Any]) -> Document:
    return Document(
        id=row.id,
        title=row.title,
        content=row.content,
        content_hash=row.content_hash,
        source=row.source,
        type=row.type,
        tags=tuple(row.tags),
        collection=row.collection,
        status=DocumentStatus(row.status),
        chunk_count=row.chunk_count,
        ingested_by_key_id=row.ingested_by_key_id,
        ingested_from_ip=row.ingested_from_ip,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
    )
