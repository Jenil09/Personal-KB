"""The persisted entities, free of SQLAlchemy.

Services and, later, the chunker and ingestion flow work with these; only
`adapters/postgres` knows they correspond to rows. Frozen, because a document
read out of Postgres is a snapshot — mutating one in place would suggest the
change had somewhere to go.

`New*` types carry what a caller supplies; `Document` and `Chunk` carry what
Postgres has, including the columns it fills in itself. Keeping them apart means
a repository signature cannot ask for an `id` and a `created_at` that the caller
has no business inventing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from ipaddress import IPv4Address, IPv6Address
from uuid import UUID

__all__ = [
    "Chunk",
    "Document",
    "DocumentFilter",
    "DocumentPage",
    "DocumentStatus",
    "IpAddress",
    "NewChunk",
    "NewDocument",
]

IpAddress = IPv4Address | IPv6Address


class DocumentStatus(StrEnum):
    """Where a document is in the two-store write (AD-003).

    `PENDING` is not a queue state — it means Postgres has the document and
    Chroma may not, and the startup reconciliation pass exists to close that
    window.
    """

    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NewDocument:
    """A document to insert, as the ingestion flow has it before the write."""

    id: UUID
    title: str
    content: str
    content_hash: str
    type: str
    collection: str
    source: str | None = None
    tags: tuple[str, ...] = ()
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    # Provenance (AD-014). Both nullable: a document ingested by the CLI over a
    # local socket has neither, and inventing values would make the trail lie.
    ingested_by_key_id: str | None = None
    ingested_from_ip: IpAddress | None = None


@dataclass(frozen=True, slots=True)
class Document:
    id: UUID
    title: str
    content: str
    content_hash: str
    type: str
    collection: str
    status: DocumentStatus
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    source: str | None = None
    tags: tuple[str, ...] = ()
    ingested_by_key_id: str | None = None
    ingested_from_ip: IpAddress | None = None
    deleted_at: datetime | None = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


@dataclass(frozen=True, slots=True)
class NewChunk:
    id: UUID
    document_id: UUID
    ordinal: int
    text: str
    text_hash: str
    token_count: int
    chroma_id: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: UUID
    document_id: UUID
    ordinal: int
    text: str
    text_hash: str
    token_count: int
    chroma_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of `GET /v1/documents`.

    `total` is a separate count query. At a ceiling of ~100 documents an exact
    count is free; if the corpus ever made it expensive the honest fix is to
    drop the field, not to approximate it.
    """

    documents: tuple[Document, ...] = ()
    total: int = 0
    limit: int = 0
    offset: int = 0


@dataclass(frozen=True, slots=True)
class DocumentFilter:
    """The filters `GET /v1/documents` and AD-005's tag lookup share.

    `tags` matches documents carrying **any** of the given tags. AD-005 rejected
    the boolean-metadata-key alternative partly because it made "any of these
    tags" awkward, so any-of is the semantics it was defending; `match_all`
    turns it into all-of for a caller that wants the intersection. Both forms
    are answerable from the GIN index.
    """

    type: str | None = None
    source: str | None = None
    collection: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    match_all_tags: bool = False
