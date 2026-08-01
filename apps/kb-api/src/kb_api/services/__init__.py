"""Application services. They depend on ports in `domain/`, never on adapters."""

from kb_api.services.ingestion import (
    IngestionService,
    IngestOutcome,
    IngestRequest,
    IngestResult,
)
from kb_api.services.providers import ProviderRegistry, ResolvedProvider
from kb_api.services.reconciliation import ReconciliationReport, ReconciliationService

__all__ = [
    "IngestOutcome",
    "IngestRequest",
    "IngestResult",
    "IngestionService",
    "ProviderRegistry",
    "ReconciliationReport",
    "ReconciliationService",
    "ResolvedProvider",
]
