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

from pydantic import BaseModel, Field, field_validator

from kb_api.services.ingestion import IngestOutcome, IngestResult

__all__ = ["IngestDocumentRequest", "IngestDocumentResponse"]

# Design §8 caps the body at 10 MB in middleware. This is the field-level
# ceiling, generous enough never to bind first: it exists so a `content` of one
# character and a `content` of ten megabytes get the same kind of answer.
_MAX_CONTENT_CHARS = 10 * 1024 * 1024


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
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Strip, drop blanks, de-duplicate, keep order.

        Order is kept rather than sorted because tags land in Chroma metadata as
        a pipe-delimited display string (AD-005), and a caller that sees its own
        tags reordered in a search result reasonably wonders what else changed.
        """
        seen: dict[str, None] = {}
        for tag in value:
            cleaned = tag.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        return tuple(seen)


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
