"""`/v1/search` — semantic search (PRD §6.2).

Same shape as the documents router: translate the wire body into a service
request, translate the result back, touch no adapter and construct no
`HTTPException`. The three failure modes worth documenting on the endpoint are
each raised somewhere specific — `422` from the provider registry for a name
nobody configured, `409` from the search service for a collection with nothing
in it, `502` from the Chroma adapter or the embedding port.

`POST` rather than `GET` for a read (PRD §6.2). A query is natural language with
filters attached, which is a body; and the tier-1 audit trail records full query
text (AD-013), so keeping it out of URLs, access logs, and proxy logs is the
narrower exposure.
"""

from http import HTTPStatus

from fastapi import APIRouter

from kb_api.api.v1.schemas import SearchRequest, SearchResponse
from kb_api.services.search import SearchFilters, SearchQuery, SearchService

__all__ = ["create_search_router"]


def create_search_router(search: SearchService) -> APIRouter:
    router = APIRouter(prefix="/search", tags=["search"])

    @router.post(
        "",
        status_code=HTTPStatus.OK,
        summary="Semantic search",
        response_model=SearchResponse,
        responses={
            HTTPStatus.OK: {"description": "Ranked results, nearest first"},
            HTTPStatus.UNAUTHORIZED: {"description": "Missing or unrecognised API key"},
            HTTPStatus.CONFLICT: {
                "description": "The provider's collection holds no documents (AD-006)"
            },
            HTTPStatus.UNPROCESSABLE_ENTITY: {
                "description": "Unknown provider, unknown filter key, or an empty query"
            },
            HTTPStatus.BAD_GATEWAY: {"description": "The embedding provider or Chroma failed"},
        },
    )
    async def search_documents(body: SearchRequest) -> SearchResponse:
        result = await search.search(
            SearchQuery(
                query=body.query,
                top_k=body.top_k,
                provider=body.provider,
                filters=SearchFilters(
                    type=body.filters.type,
                    source=body.filters.source,
                    tags=body.filters.tags,
                    match_all_tags=body.filters.match_all_tags,
                ),
            )
        )
        return SearchResponse.of(result)

    return router
