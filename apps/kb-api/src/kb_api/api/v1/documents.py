"""`/v1/documents` — ingestion (PRD §6.3).

The listing, single-get, and delete endpoints are Phase 8. This router holds the
one endpoint Phase 6 delivers, and it does what a router is supposed to do and
nothing else: translate a wire body into a service request, add the provenance
only the HTTP layer knows, and translate the result back. No adapter is touched
here, no error is constructed here, and no `HTTPException` exists anywhere in
this service — the `PlatformError` a service raises is mapped to problem+json by
the single handler `platform-fastapi` installs.

Auth is not declared on the handler. `create_app` mounts every `/v1` router
behind `require_api_key`, so a new endpoint cannot be added unauthenticated by
forgetting a decorator. `CurrentPrincipal` below re-reads the already-resolved
dependency; it costs a cache lookup, not a second comparison.
"""

from http import HTTPStatus
from ipaddress import ip_address

from fastapi import APIRouter, Request

from kb_api.api.v1.schemas import IngestDocumentRequest, IngestDocumentResponse
from kb_api.domain import IpAddress
from kb_api.services.ingestion import IngestionService, IngestRequest
from platform_fastapi import CurrentPrincipal

__all__ = ["create_documents_router"]


def create_documents_router(ingestion: IngestionService) -> APIRouter:
    """The service is bound here rather than resolved per request.

    It holds only ports and is stateless between calls, so there is nothing to
    build per request — and a `Depends` that reached into `app.state` would put
    the composition root's knowledge back into the router.
    """
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post(
        "",
        status_code=HTTPStatus.CREATED,
        summary="Ingest a document",
        response_model=IngestDocumentResponse,
        responses={
            HTTPStatus.CREATED: {"description": "Indexed, or unchanged and already indexed"},
            HTTPStatus.UNAUTHORIZED: {"description": "Missing or unrecognised API key"},
            HTTPStatus.CONTENT_TOO_LARGE: {"description": "Body over the configured limit"},
            HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "Unknown provider, or empty content"},
            HTTPStatus.BAD_GATEWAY: {"description": "The embedding provider or Chroma failed"},
        },
    )
    async def ingest_document(
        body: IngestDocumentRequest, request: Request, principal: CurrentPrincipal
    ) -> IngestDocumentResponse:
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

    return router


def _client_ip(request: Request) -> IpAddress | None:
    """The peer address, or `None` when there is not one.

    `X-Forwarded-For` is deliberately not read here. Trusting it without knowing
    the proxy in front is how an attacker writes any address they like into the
    audit trail; the proxy configuration that makes it trustworthy is Phase 9's,
    and until then the honest answer is the socket's own peer.
    """
    if request.client is None:
        return None
    try:
        return ip_address(request.client.host)
    except ValueError:
        # An ASGI transport with a non-address host — `httpx.ASGITransport`
        # uses `testclient`. The column is `INET`; a made-up value would be
        # worse than none.
        return None
