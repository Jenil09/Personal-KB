"""Application services. They depend on ports in `domain/`, never on adapters."""

from kb_api.services.ingestion import (
    IngestionService,
    IngestOutcome,
    IngestRequest,
    IngestResult,
)
from kb_api.services.providers import ProviderRegistry, ResolvedProvider
from kb_api.services.query_cache import CachedQuery, QueryEmbeddingCache, query_cache_key
from kb_api.services.reconciliation import ReconciliationReport, ReconciliationService
from kb_api.services.search import (
    SearchFilters,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchService,
)

__all__ = [
    "CachedQuery",
    "IngestOutcome",
    "IngestRequest",
    "IngestResult",
    "IngestionService",
    "ProviderRegistry",
    "QueryEmbeddingCache",
    "ReconciliationReport",
    "ReconciliationService",
    "ResolvedProvider",
    "SearchFilters",
    "SearchHit",
    "SearchQuery",
    "SearchResult",
    "SearchService",
    "query_cache_key",
]
