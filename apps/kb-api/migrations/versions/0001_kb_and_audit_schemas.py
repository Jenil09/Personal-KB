"""kb and kb_audit schemas

Everything the knowledge base persists, in one revision: `kb.documents` and
`kb.chunks` (Design §2.1), the tier-1 `kb_audit.request_logs` that `platform-db`
defines and this service inherits (AD-018), and the three tier-2 telemetry
tables (Design §2.2).

The schemas themselves are not created here. `run_migrations` creates them
before Alembic starts, because Alembic will not create the schema its own
version table lives in and the first `upgrade head` on a clean database would
otherwise fail. `downgrade` leaves them for the same reason: dropping `kb` would
take `kb.alembic_version` with it and destroy the record of what was rolled back.

Revision ID: 0001
Revises:
Create Date: 2026-07-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("collection", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("ingested_by_key_id", sa.Text(), nullable=True),
        sa.Column("ingested_from_ip", postgresql.INET(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("jsonb_typeof(tags) = 'array'", name="ck_documents_tags_is_array"),
        sa.CheckConstraint(
            "status IN ('pending', 'indexed', 'failed')", name="ck_documents_status"
        ),
        sa.CheckConstraint("length(content_hash) = 64", name="ck_documents_content_hash_length"),
        sa.PrimaryKeyConstraint("id"),
        schema="kb",
    )
    op.create_index(
        "ix_documents_status",
        "documents",
        ["status"],
        schema="kb",
        postgresql_where=sa.text("status <> 'indexed'"),
    )
    op.create_index(
        "ix_documents_tags",
        "documents",
        ["tags"],
        schema="kb",
        postgresql_using="gin",
        postgresql_ops={"tags": "jsonb_path_ops"},
    )
    op.create_index(
        "ix_documents_type",
        "documents",
        ["type"],
        schema="kb",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "uq_documents_content_hash_collection",
        "documents",
        ["content_hash", "collection"],
        unique=True,
        schema="kb",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("chroma_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("length(text_hash) = 64", name="ck_chunks_text_hash_length"),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_non_negative"),
        sa.CheckConstraint("token_count >= 0", name="ck_chunks_token_count_non_negative"),
        sa.ForeignKeyConstraint(["document_id"], ["kb.documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        schema="kb",
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], schema="kb")
    op.create_index("ix_chunks_text_hash", "chunks", ["text_hash"], schema="kb")

    # Tier 1 (AD-013, AD-018). Defined in `platform_db.audit.tables`; the
    # migration that creates it is still the service's.
    op.create_table(
        "request_logs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("request_id", sa.UUID(), nullable=False),
        sa.Column("key_id", sa.Text(), nullable=True),
        sa.Column("client_ip", postgresql.INET(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("repeat_burst", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="kb_audit",
    )
    op.create_index(
        "ix_request_logs_client_ip_created_at",
        "request_logs",
        ["client_ip", sa.literal_column("created_at DESC")],
        schema="kb_audit",
    )
    op.create_index(
        "ix_request_logs_created_at",
        "request_logs",
        [sa.literal_column("created_at DESC")],
        schema="kb_audit",
    )
    op.create_index(
        "ix_request_logs_key_id_created_at",
        "request_logs",
        ["key_id", sa.literal_column("created_at DESC")],
        schema="kb_audit",
    )
    op.create_index(
        "ix_request_logs_outcome",
        "request_logs",
        ["outcome"],
        schema="kb_audit",
        postgresql_where=sa.text("outcome <> 'success'"),
    )
    op.create_index(
        "ix_request_logs_repeat_burst",
        "request_logs",
        ["repeat_burst"],
        schema="kb_audit",
        postgresql_where=sa.text("repeat_burst"),
    )

    # Tier 2 — best-effort telemetry. A row here may never arrive at all.
    op.create_table(
        "token_usage_logs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("request_id", sa.UUID(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("token_source", sa.Text(), nullable=False),
        sa.Column("billable_characters", sa.Integer(), nullable=True),
        sa.Column("api_calls", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="kb_audit",
    )
    op.create_index(
        "ix_token_usage_logs_created_at",
        "token_usage_logs",
        [sa.literal_column("created_at DESC")],
        schema="kb_audit",
    )
    op.create_index(
        "ix_token_usage_logs_request_id", "token_usage_logs", ["request_id"], schema="kb_audit"
    )

    op.create_table(
        "ingest_logs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("request_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("collection", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("chunks_created", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("chunks_reused", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("content_bytes", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("stage_timings_ms", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="kb_audit",
    )
    op.create_index(
        "ix_ingest_logs_created_at",
        "ingest_logs",
        [sa.literal_column("created_at DESC")],
        schema="kb_audit",
    )
    op.create_index("ix_ingest_logs_document_id", "ingest_logs", ["document_id"], schema="kb_audit")

    op.create_table(
        "error_logs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("request_id", sa.UUID(), nullable=False),
        sa.Column("error_code", sa.Text(), nullable=False),
        sa.Column("exception_type", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("stack", sa.Text(), nullable=True),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="kb_audit",
    )
    op.create_index(
        "ix_error_logs_created_at",
        "error_logs",
        [sa.literal_column("created_at DESC")],
        schema="kb_audit",
    )
    op.create_index("ix_error_logs_error_code", "error_logs", ["error_code"], schema="kb_audit")


def downgrade() -> None:
    op.drop_table("error_logs", schema="kb_audit")
    op.drop_table("ingest_logs", schema="kb_audit")
    op.drop_table("token_usage_logs", schema="kb_audit")
    op.drop_table("request_logs", schema="kb_audit")
    # `chunks` before `documents`: the foreign key points that way, and the
    # cascade only covers row deletes, not the table.
    op.drop_table("chunks", schema="kb")
    op.drop_table("documents", schema="kb")
