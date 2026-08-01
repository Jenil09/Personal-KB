"""`/v1/documents` — ingest, list, read, delete (PRD §6.3-6.6).

Routers do what a router is supposed to do and nothing else: translate a wire
body into a service request, add the provenance only the HTTP layer knows, and
translate the result back. No adapter is touched here, no error is constructed
here, and no `HTTPException` exists anywhere in this service — the
`PlatformError` a service raises is mapped to problem+json by the single handler
`platform-fastapi` installs.

**Scopes are declared per route, and this prefix is why** (AD-024). Listing and
reading are `search`; ingesting and deleting are `write`. The n8n key holds
`search` alone, so the workflow that processes scraped, untrusted content can
read the corpus and cannot replace or purge it. Mounting the check on the router
would force those two halves into separate routers to say something each route
can say for itself.

Authentication is not declared at all. `create_app` mounts every `/v1` router
behind `require_api_key`, so a new endpoint cannot be added unauthenticated by
forgetting a decorator; `require_scope` composes on top of that already-resolved
dependency rather than repeating it.

Each handler calls `record_operation` before doing its work. AD-013 wants the
operation and its payload on the tier-1 row, and a request that fails halfway is
the one whose payload matters most — so it is recorded on the way in, not built
from a result that may never exist.
"""

from http import HTTPStatus
from ipaddress import ip_address
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response

from kb_api.api.v1.schemas import (
    DocumentDetail,
    DocumentListResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
)
from kb_api.domain import DocumentFilter, IpAddress
from kb_api.services.documents import DocumentService
from kb_api.services.ingestion import IngestionService, IngestRequest
from platform_fastapi import CurrentPrincipal, record_operation, require_scope

__all__ = ["create_documents_router"]

_MAX_PAGE = 100

_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: {"description": "Missing or unrecognised API key"},
    HTTPStatus.FORBIDDEN: {"description": "The key lacks the scope this route needs (AD-024)"},
    HTTPStatus.TOO_MANY_REQUESTS: {"description": "Rate limit exceeded (AD-014)"},
}


def create_documents_router(ingestion: IngestionService, documents: DocumentService) -> APIRouter:
    """The services are bound here rather than resolved per request.

    They hold only ports and are stateless between calls, so there is nothing to
    build per request — and a `Depends` that reached into `app.state` would put
    the composition root's knowledge back into the router.
    """
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post(
        "",
        status_code=HTTPStatus.CREATED,
        summary="Ingest a document",
        response_model=IngestDocumentResponse,
        dependencies=[Depends(require_scope("write"))],
        responses={
            HTTPStatus.CREATED: {"description": "Indexed, or unchanged and already indexed"},
            **_AUTH_RESPONSES,
            HTTPStatus.CONTENT_TOO_LARGE: {"description": "Body over the configured limit"},
            HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "Unknown provider, or empty content"},
            HTTPStatus.BAD_GATEWAY: {"description": "The embedding provider or Chroma failed"},
        },
    )
    async def ingest_document(
        body: IngestDocumentRequest, request: Request, principal: CurrentPrincipal
    ) -> IngestDocumentResponse:
        record_operation(
            request,
            "ingest",
            {
                "title": body.title,
                "source": body.source,
                "type": body.type,
                "tags": list(body.tags),
                "provider": body.provider,
                # The size of what was sent, not of what was stored. AD-013 wants
                # to know what the caller submitted, and normalisation happens
                # later in the flow.
                "content_bytes": len(body.content.encode()),
            },
        )
        result = await ingestion.ingest(
            IngestRequest(
                title=body.title,
                content=body.content,
                type=body.type,
                source=body.source,
                tags=body.tags,
                provider=body.provider,
                # Provenance (AD-014). Recorded from what the transport actually
                # saw, never from the body — a client that could name its own
                # key_id could name someone else's.
                ingested_by_key_id=principal.key_id,
                ingested_from_ip=_client_ip(request),
            )
        )
        return IngestDocumentResponse.of(result)

    @router.get(
        "",
        summary="List documents",
        response_model=DocumentListResponse,
        dependencies=[Depends(require_scope("search"))],
        responses={
            HTTPStatus.OK: {"description": "One page, newest first"},
            **_AUTH_RESPONSES,
        },
    )
    async def list_documents(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=_MAX_PAGE)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        # Aliased rather than named `type`: the wire name is PRD §6.4's and the
        # Python name must not shadow the builtin.
        document_type: Annotated[str | None, Query(alias="type", max_length=64)] = None,
        source: Annotated[str | None, Query(max_length=1024)] = None,
        tags: Annotated[
            list[str] | None,
            Query(description="Repeatable. Matches documents carrying any of these tags."),
        ] = None,
        match_all_tags: Annotated[
            bool, Query(description="Require every tag rather than any of them.")
        ] = False,
        collection: Annotated[str | None, Query(max_length=256)] = None,
    ) -> DocumentListResponse:
        filters = DocumentFilter(
            type=document_type,
            source=source,
            collection=collection,
            tags=tuple(tags or ()),
            match_all_tags=match_all_tags,
        )
        record_operation(
            request,
            "documents.list",
            {"limit": limit, "offset": offset, "filters": _describe(filters)},
        )
        page = await documents.list(filters, limit=limit, offset=offset)
        return DocumentListResponse.of(page)

    @router.get(
        "/{document_id}",
        summary="Get a document",
        response_model=DocumentDetail,
        dependencies=[Depends(require_scope("search"))],
        responses={
            HTTPStatus.OK: {"description": "The document, original content included"},
            **_AUTH_RESPONSES,
            HTTPStatus.NOT_FOUND: {"description": "No document with that id"},
        },
    )
    async def get_document(document_id: UUID, request: Request) -> DocumentDetail:
        record_operation(request, "documents.get", {"document_id": str(document_id)})
        return DocumentDetail.of(await documents.get(document_id))

    @router.delete(
        "/{document_id}",
        status_code=HTTPStatus.NO_CONTENT,
        summary="Delete a document",
        dependencies=[Depends(require_scope("write"))],
        responses={
            HTTPStatus.NO_CONTENT: {
                "description": (
                    "Removed from both stores. Returned whether or not the document "
                    "existed — the delete is idempotent (PRD §6.6)."
                )
            },
            **_AUTH_RESPONSES,
        },
    )
    async def delete_document(document_id: UUID, request: Request) -> Response:
        record_operation(request, "documents.delete", {"document_id": str(document_id)})
        result = await documents.delete(document_id)
        # `204` either way. The body would be the only place to report which of
        # the two happened, and a 204 has none — which is the point: a client
        # retrying a delete it already completed should not have to care.
        return Response(
            status_code=HTTPStatus.NO_CONTENT,
            headers={"X-Deleted": "true" if result.deleted else "false"},
        )

    return router


def _describe(filters: DocumentFilter) -> dict[str, object]:
    """Only the filters that were actually set, for the audit payload."""
    described: dict[str, object] = {}
    if filters.type is not None:
        described["type"] = filters.type
    if filters.source is not None:
        described["source"] = filters.source
    if filters.collection is not None:
        described["collection"] = filters.collection
    if filters.tags:
        described["tags"] = list(filters.tags)
        described["match_all_tags"] = filters.match_all_tags
    return described


def _client_ip(request: Request) -> IpAddress | None:
    """The peer address, or `None` when there is not one.

    `X-Forwarded-For` is deliberately not read here. Under AD-023 nothing
    published to the internet reaches this service, and trusting a forwarded
    header without knowing what set it is how an attacker writes any address
    they like into the provenance columns.
    """
    if request.client is None:
        return None
    try:
        return ip_address(request.client.host)
    except ValueError:
        # An ASGI transport with a non-address host. The column is `INET`; a
        # made-up value would be worse than none.
        return None
