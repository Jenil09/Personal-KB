"""Request and response bodies for `/v1` (PRD §6).

These are the API contract. They are pydantic models rather than the domain
dataclasses on purpose: the wire format has to stay stable across refactors of
the entities behind it, and OpenAPI is generated from exactly what is declared
here (AD-016). The service layer never sees one of these — the router converts.

Examples are attached because Phase 8's exit criterion is that Swagger UI is
"complete enough to drive the API without reading the source", and examples are
most of what makes that true.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kb_api.domain import Document, DocumentPage, DocumentStatus
from kb_api.services.ingestion import IngestOutcome, IngestResult
from kb_api.services.search import SearchHit, SearchResult
from kb_api.services.stats import ServiceStats

__all__ = [
    "CollectionStatsResponse",
    "DocumentDetail",
    "DocumentListResponse",
    "DocumentSummary",
    "IngestDocumentRequest",
    "IngestDocumentResponse",
    "SearchFiltersRequest",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "SearchResultMetadata",
    "StatsResponse",
    "TokenUsageResponse",
]

# Design §8 caps the body at 10 MB in middleware. This is the field-level
# ceiling, generous enough never to bind first: it exists so a `content` of one
# character and a `content` of ten megabytes get the same kind of answer.
_MAX_CONTENT_CHARS = 10 * 1024 * 1024

# A query is embedded in a single request, so the model's context window is the
# real ceiling. This is far below it and above any natural-language query — a
# caller sending more than this has a document, not a question, and the port
# would reject it after the request had already been accepted.
_MAX_QUERY_CHARS = 4096

# PRD §6.2 asks for ranked results, not a bulk export. The ceiling keeps one
# request from pulling the whole collection back through the response.
_MAX_TOP_K = 50


def _clean_tags(value: tuple[str, ...]) -> tuple[str, ...]:
    """Strip, drop blanks, de-duplicate, keep order.

    Order is kept rather than sorted because tags land in Chroma metadata as a
    pipe-delimited display string (AD-005), and a caller that sees its own tags
    reordered in a search result reasonably wonders what else changed.

    Shared by ingest and by search's filters: a tag that is cleaned one way on
    the way in and another way on the way out matches nothing, and it would do
    so silently.
    """
    seen: dict[str, None] = {}
    for tag in value:
        cleaned = tag.strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return tuple(seen)


class IngestDocumentRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Redshift Architecture",
                    "content": "# Redshift Architecture\n\nBare metal Kubernetes...",
                    "source": "redshift_architecture.md",
                    "type": "architecture",
                    "tags": ["ansible", "hardening"],
                    "provider": "openai",
                }
            ]
        }
    }

    title: Annotated[str, Field(min_length=1, max_length=512)]
    content: Annotated[str, Field(min_length=1, max_length=_MAX_CONTENT_CHARS)]
    type: Annotated[str, Field(min_length=1, max_length=64)]
    source: Annotated[str | None, Field(default=None, max_length=1024)] = None
    tags: Annotated[tuple[str, ...], Field(default=())] = ()
    provider: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Which embedding provider's collection to write to (AD-006). "
                "Defaults to the service's configured provider. Not an enum: "
                "which providers exist depends on which API keys are configured."
            ),
        ),
    ] = None

    @field_validator("tags", mode="after")
    @classmethod
    def _tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _clean_tags(value)


class IngestDocumentResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                    "chunks_created": 14,
                    "chunks_reused": 0,
                    "total_tokens": 3842,
                    "status": "success",
                    "collection": "kb__openai__text_embedding_3_small__1536__c1",
                    "superseded": [],
                }
            ]
        }
    }

    document_id: UUID
    chunks_created: int
    chunks_reused: int
    total_tokens: int
    status: IngestOutcome
    collection: str
    superseded: tuple[UUID, ...] = Field(
        default=(),
        description=(
            "Documents replaced by this ingest because they shared its `source` "
            "(AD-020). Their vectors have been purged."
        ),
    )

    @classmethod
    def of(cls, result: IngestResult) -> "IngestDocumentResponse":
        return cls(
            document_id=result.document_id,
            chunks_created=result.chunks_created,
            chunks_reused=result.chunks_reused,
            total_tokens=result.total_tokens,
            status=result.outcome,
            collection=result.collection,
            superseded=result.superseded,
        )


class SearchFiltersRequest(BaseModel):
    """PRD §6.2's `filters`.

    `extra="forbid"`, unlike every other model here. A misspelled filter key is
    the one validation failure that would otherwise be invisible: the request
    succeeds, the filter silently does not apply, and the caller gets a larger
    result set that looks entirely plausible. Being wrong quietly is worse than
    a `422` naming the key.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[str | None, Field(default=None, max_length=64)] = None
    source: Annotated[str | None, Field(default=None, max_length=1024)] = None
    tags: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Matches documents carrying any of these tags. Resolved through "
                "PostgreSQL, not Chroma metadata (AD-005)."
            ),
        ),
    ] = ()
    match_all_tags: Annotated[
        bool,
        Field(default=False, description="Require every tag rather than any of them."),
    ] = False

    @field_validator("tags", mode="after")
    @classmethod
    def _tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _clean_tags(value)


class SearchRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "bare metal kubernetes with cilium and longhorn",
                    "top_k": 5,
                    "provider": "openai",
                    "filters": {"type": "architecture", "tags": ["ansible"]},
                }
            ]
        }
    }

    query: Annotated[str, Field(min_length=1, max_length=_MAX_QUERY_CHARS)]
    top_k: Annotated[int, Field(default=5, ge=1, le=_MAX_TOP_K)] = 5
    provider: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Which provider's collection to search (AD-006). A provider whose "
                "collection holds nothing is a 409, not an empty result set."
            ),
        ),
    ] = None
    filters: SearchFiltersRequest = SearchFiltersRequest()


class SearchResultMetadata(BaseModel):
    """The chunk's provenance, read back out of Chroma's metadata (AD-004)."""

    document_id: str
    title: str
    type: str
    tags: tuple[str, ...] = ()
    source: str | None = None
    ordinal: int = 0


class SearchResultItem(BaseModel):
    id: str
    text: str
    metadata: SearchResultMetadata
    score: float = Field(description="Cosine similarity, derived as `1 - distance`.")

    @classmethod
    def of(cls, hit: SearchHit) -> "SearchResultItem":
        return cls(
            id=hit.id,
            text=hit.text,
            metadata=SearchResultMetadata(
                document_id=hit.metadata.document_id,
                title=hit.metadata.title,
                type=hit.metadata.type,
                tags=hit.metadata.tags,
                source=hit.metadata.source,
                ordinal=hit.metadata.ordinal,
            ),
            # Six places. The float carries more, and none of it is meaningful
            # against an approximate-nearest-neighbour index.
            score=round(hit.score, 6),
        )


class SearchResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301:7",
                            "text": "Bare metal Kubernetes with Cilium...",
                            "metadata": {
                                "document_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                                "title": "Redshift Architecture",
                                "type": "architecture",
                                "tags": ["ansible", "hardening"],
                                "source": "redshift_architecture.md",
                                "ordinal": 7,
                            },
                            "score": 0.872,
                        }
                    ],
                    "query_tokens": 18,
                    "latency_ms": 412,
                }
            ]
        }
    }

    results: tuple[SearchResultItem, ...]
    query_tokens: int = Field(
        description=(
            "Tokens the provider counted for this query. A cache hit reports "
            "the count from when the query was embedded, so repeating a search "
            "does not change its answer; it is zero only when the query was "
            "never sent — a tag filter that matched no documents."
        )
    )
    latency_ms: int

    @classmethod
    def of(cls, result: SearchResult) -> "SearchResponse":
        return cls(
            results=tuple(SearchResultItem.of(hit) for hit in result.hits),
            query_tokens=result.query_tokens,
            latency_ms=result.latency_ms,
        )


class DocumentSummary(BaseModel):
    """A document as it appears in a listing — everything but the content.

    `content` is deliberately absent. The corpus averages ~0.5 MB a file, so a
    fifty-item page carrying content would be a 25 MB response to a request that
    asked "what is in here"; PRD §6.5 is the endpoint that returns content, one
    document at a time.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                    "title": "Redshift Architecture",
                    "type": "architecture",
                    "source": "redshift_architecture.md",
                    "tags": ["ansible", "hardening"],
                    "collection": "kb__openai__text_embedding_3_small__1536__c1",
                    "status": "indexed",
                    "chunk_count": 14,
                    "content_hash": "9f2c...",
                    "created_at": "2026-07-30T09:12:44Z",
                    "updated_at": "2026-07-30T09:12:47Z",
                }
            ]
        }
    }

    id: UUID
    title: str
    type: str
    collection: str
    status: DocumentStatus
    chunk_count: int
    content_hash: str
    created_at: datetime
    updated_at: datetime
    source: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def of(cls, document: Document) -> "DocumentSummary":
        return cls(
            id=document.id,
            title=document.title,
            type=document.type,
            collection=document.collection,
            status=document.status,
            chunk_count=document.chunk_count,
            content_hash=document.content_hash,
            created_at=document.created_at,
            updated_at=document.updated_at,
            source=document.source,
            tags=document.tags,
        )


class DocumentDetail(DocumentSummary):
    """PRD §6.5 — the summary plus the original content.

    Provenance (AD-014) is on this model and not on the summary: `ingested_by_key_id`
    and `ingested_from_ip` answer a forensic question about one document, and
    repeating them down a fifty-row listing invites reading the listing as an
    audit trail. The trail is `kb_audit.request_logs`.
    """

    content: str
    ingested_by_key_id: str | None = None
    ingested_from_ip: str | None = None

    @classmethod
    def of(cls, document: Document) -> "DocumentDetail":
        summary = DocumentSummary.of(document)
        return cls(
            **summary.model_dump(),
            content=document.content,
            ingested_by_key_id=document.ingested_by_key_id,
            ingested_from_ip=(
                str(document.ingested_from_ip) if document.ingested_from_ip is not None else None
            ),
        )


class DocumentListResponse(BaseModel):
    """One page of PRD §6.4.

    `total` is the count matching the filters, not the page size — a client
    paginating needs to know when to stop, and deriving that from a short page
    breaks when the last page happens to be full.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [{"documents": [], "total": 24, "limit": 50, "offset": 0}]
        }
    }

    documents: tuple[DocumentSummary, ...]
    total: int
    limit: int
    offset: int

    @classmethod
    def of(cls, page: DocumentPage) -> "DocumentListResponse":
        return cls(
            documents=tuple(DocumentSummary.of(document) for document in page.documents),
            total=page.total,
            limit=page.limit,
            offset=page.offset,
        )


class CollectionStatsResponse(BaseModel):
    """One collection, as PRD §6.7's "collection information"."""

    name: str
    provider: str
    model: str
    dimensions: int
    vectors: int | None = Field(
        default=None,
        description=(
            "Vectors the collection holds. `null` means Chroma could not be "
            "reached — distinct from `0`, which means the collection is empty."
        ),
    )


class TokenUsageResponse(BaseModel):
    """Cumulative embedding spend over the reporting window.

    Exact and estimated counts are separate fields rather than one total (AD-017).
    Gemini reports no token count for embeddings, so a combined figure would look
    like a billing number while containing an estimate.
    """

    exact_tokens: int
    estimated_tokens: int
    api_calls: int
    window_days: int


class StatsResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents_by_status": {"indexed": 24},
                    "documents_by_collection": {"kb__openai__text_embedding_3_small__1536__c1": 24},
                    "total_documents": 24,
                    "total_chunks": 6700,
                    "total_tokens_stored": 2_980_000,
                    "collections": [
                        {
                            "name": "kb__openai__text_embedding_3_small__1536__c1",
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "dimensions": 1536,
                            "vectors": 6700,
                        }
                    ],
                    "tokens": {
                        "exact_tokens": 2_980_000,
                        "estimated_tokens": 0,
                        "api_calls": 68,
                        "window_days": 30,
                    },
                    "telemetry_dropped": 0,
                    "telemetry_written": 412,
                    "telemetry_queue_depth": 0,
                    "audit_spill_depth": 0,
                    "recent_bursts": 0,
                }
            ]
        }
    }

    documents_by_status: dict[str, int]
    documents_by_collection: dict[str, int]
    total_documents: int
    total_chunks: int
    total_tokens_stored: int
    collections: tuple[CollectionStatsResponse, ...]
    tokens: TokenUsageResponse
    telemetry_dropped: int = Field(
        description="Tier-2 records lost to a full queue. Non-zero means telemetry is incomplete."
    )
    telemetry_written: int
    telemetry_queue_depth: int
    audit_spill_depth: int = Field(
        description=(
            "Tier-1 records written to the spill file and not yet reconciled "
            "(AD-013). Non-zero also reports `degraded` on `/health`."
        )
    )
    recent_bursts: int = Field(
        description="Requests flagged as an identical-query burst in the last 7 days (AD-014)."
    )

    @classmethod
    def of(cls, stats: ServiceStats, *, token_window_days: int = 30) -> "StatsResponse":
        return cls(
            documents_by_status=dict(stats.corpus.documents_by_status),
            documents_by_collection=dict(stats.corpus.documents_by_collection),
            total_documents=sum(stats.corpus.documents_by_status.values()),
            total_chunks=stats.corpus.total_chunks,
            total_tokens_stored=stats.corpus.total_tokens_stored,
            collections=tuple(
                CollectionStatsResponse(
                    name=collection.name,
                    provider=collection.provider,
                    model=collection.model,
                    dimensions=collection.dimensions,
                    vectors=collection.vectors,
                )
                for collection in stats.collections
            ),
            tokens=TokenUsageResponse(
                exact_tokens=stats.tokens.exact_tokens,
                estimated_tokens=stats.tokens.estimated_tokens,
                api_calls=stats.tokens.api_calls,
                window_days=token_window_days,
            ),
            telemetry_dropped=stats.telemetry_dropped,
            telemetry_written=stats.telemetry_written,
            telemetry_queue_depth=stats.telemetry_queue_depth,
            audit_spill_depth=stats.audit_spill_depth,
            recent_bursts=stats.recent_bursts,
        )
