"""Postgres — the system of record (AD-003).

`kb_metadata` and `telemetry_metadata` are what `migrations/env.py` hands
Alembic, alongside `platform_db.audit_metadata` for the tier-1 table the library
owns (AD-018).
"""

from kb_api.adapters.postgres.chunks import ChunkRepository
from kb_api.adapters.postgres.documents import DocumentRepository
from kb_api.adapters.postgres.tables import KB_SCHEMA, chunks, documents, kb_metadata
from kb_api.adapters.postgres.telemetry_tables import (
    error_logs,
    ingest_logs,
    telemetry_metadata,
    token_usage_logs,
)

__all__ = [
    "KB_SCHEMA",
    "ChunkRepository",
    "DocumentRepository",
    "chunks",
    "documents",
    "error_logs",
    "ingest_logs",
    "kb_metadata",
    "telemetry_metadata",
    "token_usage_logs",
]
