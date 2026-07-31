"""`create_app` as a whole — Phase 2's exit criterion.

"A ten-line service using only this lib boots, enforces auth, returns
structured errors, and reports health."
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from pydantic import SecretStr

from platform_core import NotFoundError
from platform_fastapi import CheckResult, HealthCheck, HttpServiceSettings, create_app


@pytest.fixture
def service() -> FastAPI:
    """The whole ten lines."""
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.get("/{document_id}")
    async def get_document(document_id: int) -> dict[str, int]:
        if document_id != 1:
            raise NotFoundError(f"document {document_id} does not exist")
        return {"id": document_id}

    async def postgres() -> CheckResult:
        return CheckResult.ok()

    settings = HttpServiceSettings(
        service_name="example", service_version="1.2", api_keys={"n8n": SecretStr("key")}
    )
    return create_app(
        settings,
        routers=[router],
        health_checks=[HealthCheck(name="postgres", probe=postgres)],
    )


@pytest.fixture
async def client(service: FastAPI, client_for) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(service) as http_client:
        yield http_client


async def test_it_reports_health_without_a_key(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.2", "postgres": "connected"}


async def test_it_enforces_auth(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/documents/1")).status_code == 401


async def test_it_serves_the_route_under_v1(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/documents/1", headers={"Authorization": "Bearer key"})

    assert response.status_code == 200
    assert response.json() == {"id": 1}


async def test_it_returns_structured_errors(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/documents/7", headers={"Authorization": "Bearer key"})

    assert response.status_code == 404
    assert response.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "document 7 does not exist",
        "instance": "/v1/documents/7",
        "code": "not_found",
        "request_id": response.headers["X-Request-ID"],
    }


async def test_the_api_is_documented(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()

    assert schema["openapi"].startswith("3.1")
    assert schema["info"] == {"title": "example", "version": "1.2"}
    assert "/v1/documents/{document_id}" in schema["paths"]
    assert (await client.get("/docs")).status_code == 200


async def test_lifespan_runs_around_the_app(settings, client_for) -> None:
    """Driven explicitly: `ASGITransport` sends requests but no lifespan events."""
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        events.append("startup")
        yield
        events.append("shutdown")

    app = create_app(settings, lifespan=lifespan)
    async with app.router.lifespan_context(app):
        assert events == ["startup"]
        async with client_for(app) as client:
            assert (await client.get("/health")).status_code == 200

    assert events == ["startup", "shutdown"]


async def test_cors_is_absent_unless_configured(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"Origin": "https://n8n.example"})

    assert "access-control-allow-origin" not in response.headers


async def test_cors_is_applied_when_origins_are_configured(make_app, client_for) -> None:
    app = make_app(cors_origins=("https://n8n.example",))

    async with client_for(app) as client:
        preflight = await client.options(
            "/health",
            headers={
                "Origin": "https://n8n.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        actual = await client.get("/health", headers={"Origin": "https://n8n.example"})

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://n8n.example"
    # The correlation ID is only useful to a browser client if it is exposed.
    assert "X-Request-ID" in actual.headers["access-control-expose-headers"]


async def test_cors_headers_survive_an_error_response(make_app, client_for) -> None:
    """CORS sits outside the request context middleware, so a browser can read
    the problem body rather than a bare network error."""
    router = APIRouter()

    @router.get("/boom")
    async def boom() -> None:
        raise RuntimeError("unhandled")

    app = make_app(routers=[router], cors_origins=("https://n8n.example",))

    async with client_for(app) as client:
        response = await client.get(
            "/v1/boom",
            headers={"Authorization": "Bearer n8n-secret-value", "Origin": "https://n8n.example"},
        )

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "https://n8n.example"
