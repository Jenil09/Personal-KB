"""Async SQLAlchemy engine, Alembic wiring, and the two-tier audit trail."""

from platform_db.audit import (
    AUDIT_SCHEMA,
    AuditRecord,
    AuditTrail,
    Outcome,
    SpillFile,
    TelemetryEvent,
    TelemetrySink,
    activity_by_ip,
    activity_by_key,
    audit_metadata,
    failures_in_window,
    fingerprint_credential,
    ingests_by_key,
    repeat_bursts,
    request_logs,
    traffic_summary,
)
from platform_db.engine import Database
from platform_db.migrations import run_migrations
from platform_db.settings import AuditSettings, DatabaseSettings

__all__ = [
    "AUDIT_SCHEMA",
    "AuditRecord",
    "AuditSettings",
    "AuditTrail",
    "Database",
    "DatabaseSettings",
    "Outcome",
    "SpillFile",
    "TelemetryEvent",
    "TelemetrySink",
    "__version__",
    "activity_by_ip",
    "activity_by_key",
    "audit_metadata",
    "failures_in_window",
    "fingerprint_credential",
    "ingests_by_key",
    "repeat_bursts",
    "request_logs",
    "run_migrations",
    "traffic_summary",
]

__version__ = "0.1.0"
