"""The `/v1` responses, as this client parses them.

Restated rather than imported from `kb_api.api.v1.schemas`. The import would
work today — one `uv.lock`, one `.venv` — and it would be the wrong dependency:
`kb-cli` is installed on a laptop with `uv tool install` and `kb-mcp` is deployed
from its own image, and both talk to a service deployed from a different commit,
so they must parse *the contract*, not whatever the service's source happens to
say this week. A client that cannot be older than its server is not a client.

`extra="ignore"` throughout, which is the half of that argument that has teeth:
a service that adds a response field must not break every installed copy of
every consumer. Fields are only added here when a consumer has something to do
with them.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CollectionStats",
    "DocumentDetail",
    "DocumentPage",
    "DocumentSummary",
    "IngestResult",
    "SearchHit",
    "SearchHitMetadata",
    "SearchResponse",
    "Stats",
    "TokenUsage",
]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class IngestOutcome(StrEnum):
    SUCCESS = "success"
    UNCHANGED = "unchanged"


class DocumentSummary(_Wire):
    id: UUID
    title: str
    type: str
    collection: str
    status: str
    chunk_count: int
    content_hash: str
    created_at: datetime
    updated_at: datetime
    source: str | None = None
    tags: tuple[str, ...] = ()


class DocumentDetail(DocumentSummary):
    content: str
    ingested_by_key_id: str | None = None
    ingested_from_ip: str | None = None


class DocumentPage(_Wire):
    documents: tuple[DocumentSummary, ...]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.documents) < self.total


class IngestResult(_Wire):
    document_id: UUID
    chunks_created: int
    chunks_reused: int
    total_tokens: int
    status: str
    collection: str
    superseded: tuple[UUID, ...] = ()

    @property
    def unchanged(self) -> bool:
        """A re-ingest of identical content, which embedded nothing (AD-008).

        Worth surfacing distinctly in a directory walk: "40 files, 3 changed" is
        the useful summary, and it is the difference between a re-run that cost
        nothing and one that cost a re-embed.
        """
        return self.status == IngestOutcome.UNCHANGED


class SearchHitMetadata(_Wire):
    document_id: str
    title: str
    type: str
    tags: tuple[str, ...] = ()
    source: str | None = None
    ordinal: int = 0


class SearchHit(_Wire):
    id: str
    text: str
    metadata: SearchHitMetadata
    score: float


class SearchResponse(_Wire):
    results: tuple[SearchHit, ...]
    query_tokens: int = 0
    latency_ms: int = 0


class CollectionStats(_Wire):
    name: str
    provider: str
    model: str
    dimensions: int
    vectors: int | None = None


class TokenUsage(_Wire):
    exact_tokens: int = 0
    estimated_tokens: int = 0
    api_calls: int = 0
    window_days: int = 30


class Stats(_Wire):
    documents_by_status: dict[str, int] = Field(default_factory=dict)
    documents_by_collection: dict[str, int] = Field(default_factory=dict)
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens_stored: int = 0
    collections: tuple[CollectionStats, ...] = ()
    tokens: TokenUsage = TokenUsage()
    telemetry_dropped: int = 0
    telemetry_written: int = 0
    telemetry_queue_depth: int = 0
    audit_spill_depth: int = 0
    recent_bursts: int = 0

    @property
    def degraded(self) -> bool:
        """What `/health` calls `degraded` — a spilled tier-1 audit row (AD-013).

        Surfaced by the status view because it is the one number here that means
        something is wrong right now rather than describing the corpus.
        """
        return self.audit_spill_depth > 0
