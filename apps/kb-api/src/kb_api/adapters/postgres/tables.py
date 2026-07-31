"""Schema `kb` — the system of record (AD-003).

Postgres holds the original content, so Chroma can be dropped and rebuilt from
here at any time. That is the whole reason `documents.content` exists as a
column rather than only as a stream of chunks.

Core `Table` objects rather than the declarative ORM, matching
`platform_db.audit.tables`. The repositories issue statements and map rows onto
frozen domain entities; there is no identity map or lazy loading to want, and
Core keeps the SQL in the file where the indexes it depends on are declared.

Every index here backs a query in `repositories.py`. Adding one without a query,
or a query without one, is how a personal-scale table quietly starts sequentially
scanning.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID

__all__ = ["KB_SCHEMA", "chunks", "documents", "kb_metadata"]

KB_SCHEMA = "kb"

kb_metadata = MetaData(schema=KB_SCHEMA)

documents = Table(
    "documents",
    kb_metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("title", Text, nullable=False),
    # The original, normalised. Re-embedding under a new model or a new chunker
    # reads from this column and nothing else (AD-003).
    Column("content", Text, nullable=False),
    # sha256 of the normalised content — the idempotency key (AD-008).
    Column("content_hash", Text, nullable=False),
    Column("source", Text),
    Column("type", Text, nullable=False),
    Column("tags", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    # Which Chroma collection holds this document's vectors. Part of the
    # identity of the row: the same content embedded under a second provider is
    # a second document, not an update (AD-006).
    Column("collection", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("chunk_count", Integer, nullable=False, server_default=text("0")),
    # Provenance (AD-014): what this service honestly offers against a malicious
    # payload is tracing it back to a submitter, not preventing it.
    Column("ingested_by_key_id", Text),
    Column("ingested_from_ip", INET),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    # Soft delete is the crash-safety marker: it is set before the Chroma purge,
    # so a crash between the two leaves a row that is already unsearchable
    # rather than one that is gone from Postgres and still in the index.
    Column("deleted_at", DateTime(timezone=True)),
    CheckConstraint(
        "status IN ('pending', 'indexed', 'failed')",
        name="ck_documents_status",
    ),
    CheckConstraint("jsonb_typeof(tags) = 'array'", name="ck_documents_tags_is_array"),
    # Design §2.1 spells the hashes `CHAR(64)`. `TEXT` plus a length check gives
    # the same guarantee without `bpchar`'s blank-padding comparison rules,
    # which silently make 'abc' and 'abc   ' equal.
    CheckConstraint("length(content_hash) = 64", name="ck_documents_content_hash_length"),
    # Partial, so re-ingesting content that was deleted and resubmitted is
    # allowed rather than a constraint violation.
    Index(
        "uq_documents_content_hash_collection",
        "content_hash",
        "collection",
        unique=True,
        postgresql_where=text("deleted_at IS NULL"),
    ),
    # jsonb_path_ops: half the size of the default opclass and faster, at the
    # cost of only supporting `@>` — which is the only operator AD-005's tag
    # lookup uses.
    #
    # The operator class goes in `postgresql_ops` rather than inline in a
    # `text("tags jsonb_path_ops")` expression. Both emit the same DDL, but
    # Alembic cannot compare an expression index carrying an operator clause —
    # it warns and skips it — so the inline form would sit outside the drift
    # check for good.
    Index(
        "ix_documents_tags",
        "tags",
        postgresql_using="gin",
        postgresql_ops={"tags": "jsonb_path_ops"},
    ),
    Index("ix_documents_type", "type", postgresql_where=text("deleted_at IS NULL")),
    # Partial on the rare values: the startup reconciliation pass asks for
    # documents stuck in `pending`, and almost every row is `indexed`, so
    # indexing the common value would be indexing the whole table.
    Index("ix_documents_status", "status", postgresql_where=text("status <> 'indexed'")),
)

chunks = Table(
    "chunks",
    kb_metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "document_id",
        UUID(as_uuid=True),
        ForeignKey(f"{KB_SCHEMA}.documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    # sha256(normalised chunk text + model_id) — what decides whether a chunk is
    # carried forward or re-embedded (AD-008).
    Column("text_hash", Text, nullable=False),
    Column("token_count", Integer, nullable=False),
    # The Chroma-side primary key. Replaying an upsert with the same value is
    # what makes a crash between the Postgres commit and the Chroma write safe
    # to retry.
    Column("chroma_id", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
    CheckConstraint("length(text_hash) = 64", name="ck_chunks_text_hash_length"),
    CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_non_negative"),
    CheckConstraint("token_count >= 0", name="ck_chunks_token_count_non_negative"),
    Index("ix_chunks_text_hash", "text_hash"),
    # The cascade covers deletes, but the reads — every chunk of one document,
    # in order — need their own index; a foreign key does not create one.
    Index("ix_chunks_document_id", "document_id"),
)
