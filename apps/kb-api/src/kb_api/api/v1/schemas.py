"""Request and response bodies for `/v1` (PRD §6).

These are the API contract. They are pydantic models rather than the domain
dataclasses on purpose: the wire format has to stay stable across refactors of
the entities behind it, and OpenAPI is generated from exactly what is declared
here (AD-016). The service layer never sees one of these — the router converts.

Examples are attached because Phase 8's exit criterion is that Swagger UI is
"complete enough to drive the API without reading the source", and examples are
most of what makes that true.
"""

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kb_api.services.ingestion import IngestOutcome, IngestResult
from kb_api.services.search import SearchHit, SearchResult

__all__ = [
    "IngestDocumentRequest",
    "IngestDocumentResponse",
    "SearchFiltersRequest",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "SearchResultMetadata",
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
