"""Tier-2 emission — the adapter that knows which table each event belongs in.

`TelemetrySink` deliberately names no tables (AD-013): tier-2 shapes are the
service's business, so the sink takes a `Table` and this is the thing that
supplies it. Keeping that mapping here rather than in the services is what lets
`services/` depend on a port with three verbs on it instead of importing
`telemetry_tables`.

Nothing here awaits. `emit` is synchronous by design — a caller emitting
telemetry is in the middle of serving a request and must not be able to wait on
the sink, however briefly — so every method returns as soon as the record is
queued or dropped.

`created_at` is stamped by the caller's clock at emission, not by a server
default. A batch drained thirty seconds later must keep the time of the thing it
describes, for the same reason tier 1 stamps at the request.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from kb_api.adapters.postgres.telemetry_tables import error_logs, ingest_logs, token_usage_logs
from platform_db import TelemetryEvent, TelemetrySink

__all__ = ["TelemetryRecorder"]


class TelemetryRecorder:
    """Binds the tier-2 verbs the flows call to the tables they land in."""

    def __init__(self, sink: TelemetrySink) -> None:
        self._sink = sink

    def tokens_used(
        self,
        *,
        request_id: UUID,
        provider: str,
        model: str,
        operation: str,
        input_tokens: int,
        token_source: str,
        api_calls: int = 1,
        billable_characters: int | None = None,
    ) -> None:
        """One embedding call's cost (AD-017).

        `token_source` travels with the number rather than being inferred from
        the provider name, so an estimate is never quietly summed into a billing
        figure by a reader who did not know which provider produced it.
        """
        self._emit(
            token_usage_logs,
            {
                "request_id": request_id,
                "provider": provider,
                "model": model,
                "operation": operation,
                "input_tokens": input_tokens,
                "token_source": token_source,
                "api_calls": api_calls,
                "billable_characters": billable_characters,
            },
        )

    def ingest_completed(
        self,
        *,
        request_id: UUID,
        collection: str,
        outcome: str,
        document_id: UUID | None = None,
        chunks_created: int = 0,
        chunks_reused: int = 0,
        content_bytes: int | None = None,
        duration_ms: int | None = None,
        stage_timings_ms: Mapping[str, int] | None = None,
    ) -> None:
        """The pair that makes AD-008 auditable: created against reused."""
        self._emit(
            ingest_logs,
            {
                "request_id": request_id,
                "document_id": document_id,
                "collection": collection,
                "outcome": outcome,
                "chunks_created": chunks_created,
                "chunks_reused": chunks_reused,
                "content_bytes": content_bytes,
                "duration_ms": duration_ms,
                "stage_timings_ms": dict(stage_timings_ms) if stage_timings_ms else None,
            },
        )

    def error_occurred(
        self,
        *,
        request_id: UUID,
        error_code: str,
        exception_type: str,
        message: str,
        stack: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """The detail that never goes in a response (Design §5).

        The tier-1 row records *that* a request failed and with which code; this
        records why. The split matters because tier 1 is the trail that cannot be
        dropped and a stack trace is bulk — putting them in one row would make
        the guaranteed write proportional to the size of the failure.
        """
        self._emit(
            error_logs,
            {
                "request_id": request_id,
                "error_code": error_code,
                "exception_type": exception_type,
                "message": message,
                "stack": stack,
                "context": dict(context) if context else None,
            },
        )

    def _emit(self, table: Any, values: dict[str, Any]) -> None:
        values["created_at"] = datetime.now(UTC)
        self._sink.emit(TelemetryEvent(table=table, values=values))
