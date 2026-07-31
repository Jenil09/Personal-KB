"""The tier-1 table, defined once and shared.

`request_logs` lives here rather than in a service because AD-013's guarantee is
a platform property: the writer, the spill format, and the columns have to agree,
and a service redefining the table is how they stop agreeing. Services import
`audit_metadata` into their Alembic target so the migration that creates the
schema is still theirs.

Tier-2 tables are *not* here. They are per-service telemetry shapes, so the sink
takes whatever `Table` it is handed instead of naming them.

Indexes are the forensic queries in column order: what did this key do, what came
from this address, what failed, what looped.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID

__all__ = ["AUDIT_SCHEMA", "audit_metadata", "request_logs"]

AUDIT_SCHEMA = "kb_audit"

audit_metadata = MetaData(schema=AUDIT_SCHEMA)

request_logs = Table(
    "request_logs",
    audit_metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("request_id", UUID(as_uuid=True), nullable=False),
    Column("key_id", Text),  # NULL on failed auth
    Column("client_ip", INET),
    Column("user_agent", Text),
    Column("method", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("outcome", Text, nullable=False),
    Column("error_code", Text),
    Column("latency_ms", Integer, nullable=False),
    Column("operation", Text),
    Column("payload", JSONB),
    Column("repeat_burst", Boolean, nullable=False, server_default=text("false")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_request_logs_created_at", text("created_at DESC")),
    Index("ix_request_logs_key_id_created_at", "key_id", text("created_at DESC")),
    Index("ix_request_logs_client_ip_created_at", "client_ip", text("created_at DESC")),
    Index(
        "ix_request_logs_outcome",
        "outcome",
        postgresql_where=text("outcome <> 'success'"),
    ),
    Index(
        "ix_request_logs_repeat_burst",
        "repeat_burst",
        postgresql_where=text("repeat_burst"),
    ),
)
