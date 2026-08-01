"""The tier-1 record: one row per request, whatever happens (AD-013).

`created_at` is stamped here rather than defaulted to `now()` in Postgres. A
record that spends an hour in the spill file must keep the time of the request,
not the time the reconciler got to it — otherwise the trail's ordering silently
lies about exactly the incident it exists to reconstruct.
"""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, IPvAnyAddress

__all__ = ["AuditRecord", "Outcome", "fingerprint_credential"]

_FINGERPRINT_CHARS = 8


class Outcome(StrEnum):
    """Why the request ended the way it did — the forensic filter (Design §2.2)."""

    SUCCESS = "success"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"


class AuditRecord(BaseModel):
    """A `kb_audit.request_logs` row, in flight.

    Serialised to JSON for the spill file and to a parameter dict for the
    insert, so the two paths cannot drift apart.
    """

    model_config = {"frozen": True}

    request_id: UUID
    method: str
    path: str
    status_code: int
    outcome: Outcome
    latency_ms: int

    key_id: str | None = None
    client_ip: IPvAnyAddress | None = None
    user_agent: str | None = None
    error_code: str | None = None
    operation: str | None = None
    # `Any` because the payload is the operation's own shape — a search's query
    # and filters, an ingest's title, source, content_hash, and byte size — and
    # lands in a JSONB column unmodified.
    payload: dict[str, Any] | None = None
    repeat_burst: bool = False
    anomaly: bool = False
    # AD-023: the tailnet identity behind the proxy hop, when there is one.
    tailnet_user: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_row(self) -> dict[str, Any]:
        """Insert parameters. UUID, IP, and enum go across as their native types."""
        row = self.model_dump()
        # asyncpg wants a string for INET; pydantic hands back an IPv4Address.
        if self.client_ip is not None:
            row["client_ip"] = str(self.client_ip)
        row["outcome"] = self.outcome.value
        return row


def fingerprint_credential(secret: str) -> str:
    """Identify a rejected key without storing it (Design §2.2).

    Enough to see the same bad credential retried a thousand times; not enough
    to recover it, and not enough to confirm a guess without the audit trail
    already being readable.
    """
    return hashlib.sha256(secret.encode()).hexdigest()[:_FINGERPRINT_CHARS]
