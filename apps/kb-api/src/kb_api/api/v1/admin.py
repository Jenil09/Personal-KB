"""`/v1/admin/stats` — PRD §6.7.

Requires the `write` scope rather than a scope of its own (AD-024). Two scopes
cover two callers: the n8n key searches, the operator key does everything else,
and inventing an `admin` scope now would be a third name that exactly one key
would ever hold. When a third consumer appears with a different shape, that is
the information that decides whether this needs its own scope — not a guess made
before it exists.

The endpoint is a read, but it is not a cheap one to leave open: it reports
collection names, corpus size, and how much of the observability has been lost,
which together describe the deployment rather than its contents.
"""

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from kb_api.api.v1.schemas import StatsResponse
from kb_api.services.stats import StatsService
from platform_fastapi import record_operation, require_scope

__all__ = ["create_admin_router"]


def create_admin_router(stats: StatsService) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.get(
        "/stats",
        summary="Service and corpus statistics",
        response_model=StatsResponse,
        dependencies=[Depends(require_scope("write"))],
        responses={
            HTTPStatus.OK: {"description": "Counts, collections, and observability health"},
            HTTPStatus.UNAUTHORIZED: {"description": "Missing or unrecognised API key"},
            HTTPStatus.FORBIDDEN: {"description": "The key lacks the `write` scope (AD-024)"},
            HTTPStatus.TOO_MANY_REQUESTS: {"description": "Rate limit exceeded (AD-014)"},
        },
    )
    async def service_stats(request: Request) -> StatsResponse:
        record_operation(request, "admin.stats")
        return StatsResponse.of(await stats.collect())

    return router
