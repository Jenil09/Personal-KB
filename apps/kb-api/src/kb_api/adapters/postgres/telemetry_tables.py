"""Tier-2 telemetry tables — best-effort, batched, droppable (AD-013).

These live in `kb_audit` alongside `request_logs` but are owned by the service,
not by `platform-db`: the tier-2 shapes are whatever a service finds worth
measuring, which is why `TelemetrySink` takes a `Table` instead of naming one.
`request_logs` is the opposite case and stays in the library (AD-018).

They get their own `MetaData` rather than being attached to the library's
`audit_metadata`. Same schema, separate object — a service reaching into a
library's metadata to register tables makes the library's own migration story
depend on which services happen to be imported. Alembic takes a list of
metadata, so nothing is lost.

The durability difference is the thing to keep in mind reading these: a row here
may simply never arrive, because the queue was full and the sink counted a drop.
Nothing that has to be true — attribution, provenance, the security trail — is
recorded here. It goes in `request_logs`.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from platform_db import AUDIT_SCHEMA

__all__ = [
    "error_logs",
    "ingest_logs",
    "telemetry_metadata",
    "token_usage_logs",
]

telemetry_metadata = MetaData(schema=AUDIT_SCHEMA)

token_usage_logs = Table(
    "token_usage_logs",
    telemetry_metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("request_id", UUID(as_uuid=True), nullable=False),
    Column("provider", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("operation", Text, nullable=False),  # ingest | search
    Column("input_tokens", Integer, nullable=False),
    # AD-017: Gemini reports no token count for embeddings, only a billable
    # character count. This column says where the number came from, so an
    # estimate is never quietly summed into a billing figure.
    Column("token_source", Text, nullable=False),  # exact | estimated
    Column("billable_characters", Integer),
    Column("api_calls", Integer, nullable=False, server_default=text("1")),
    # Cost is derived and provider pricing changes, so it is recorded as
    # observed at the time rather than recomputed later from a rate table that
    # has since moved.
    Column("estimated_cost_usd", Numeric(12, 6)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_token_usage_logs_created_at", text("created_at DESC")),
    Index("ix_token_usage_logs_request_id", "request_id"),
)

ingest_logs = Table(
    "ingest_logs",
    telemetry_metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("request_id", UUID(as_uuid=True), nullable=False),
    Column("document_id", UUID(as_uuid=True)),
    Column("collection", Text, nullable=False),
    Column("outcome", Text, nullable=False),  # success | unchanged | failed
    # The pair that makes AD-008 auditable after the fact: how much of a
    # re-ingest was carried forward rather than re-embedded.
    Column("chunks_created", Integer, nullable=False, server_default=text("0")),
    Column("chunks_reused", Integer, nullable=False, server_default=text("0")),
    Column("content_bytes", Integer),
    Column("duration_ms", Integer),
    # Per-stage timings — chunking, embedding, the Postgres write, the Chroma
    # upsert. JSONB because the stages are the ingestion flow's business and
    # will change with it; a column per stage would migrate on every change.
    Column("stage_timings_ms", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_ingest_logs_created_at", text("created_at DESC")),
    Index("ix_ingest_logs_document_id", "document_id"),
)

error_logs = Table(
    "error_logs",
    telemetry_metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("request_id", UUID(as_uuid=True), nullable=False),
    Column("error_code", Text, nullable=False),
    Column("exception_type", Text, nullable=False),
    Column("message", Text, nullable=False),
    # The stack detail that never goes in a response (Design §5). The tier-1 row
    # records that the request failed and with which code; this records why.
    Column("stack", Text),
    Column("context", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_error_logs_created_at", text("created_at DESC")),
    Index("ix_error_logs_error_code", "error_code"),
)
