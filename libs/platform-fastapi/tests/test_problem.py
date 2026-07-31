"""Every failure leaves as RFC 9457 problem+json — Technical Design §5."""

import httpx
import pytest
from fastapi import APIRouter
from pydantic import BaseModel

from platform_core import ConflictError, NotFoundError, PlatformError, UpstreamError
from platform_fastapi import PROBLEM_CONTENT_TYPE


class Payload(BaseModel):
    limit: int


@pytest.fixture
def failing() -> APIRouter:
    router = APIRouter()

    @router.get("/not-found")
    async def not_found() -> None:
        raise NotFoundError("document 7 does not exist")

    @router.get("/conflict")
    async def conflict() -> None:
        raise ConflictError(
            "collection is not populated",
            context={"provider": "gemini", "collection": "kb_gemini_v1"},
        )

    @router.get("/upstream")
    async def upstream() -> None:
        raise UpstreamError("embedding provider timed out")

    @router.get("/base")
    async def base() -> None:
        raise PlatformError("something in the middle went wrong")

    @router.get("/context-cannot-shadow")
    async def context_cannot_shadow() -> None:
        raise NotFoundError("gone", context={"status": 200, "code": "spoofed", "kept": True})

    @router.get("/boom")
    async def boom() -> None:
        raise RuntimeError("connection string is postgres://user:hunter2@db/kb")

    @router.post("/echo")
    async def echo(payload: Payload) -> Payload:
        return payload

    return router


@pytest.fixture
async def client(make_app, client_for, failing):
    async with client_for(make_app(routers=[failing])) as http_client:
        yield http_client


@pytest.fixture
def auth(valid_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {valid_key}"}


@pytest.mark.parametrize(
    ("path", "status", "code", "title"),
    [
        ("/v1/not-found", 404, "not_found", "Not Found"),
        ("/v1/conflict", 409, "conflict", "Conflict"),
        ("/v1/upstream", 502, "upstream_error", "Bad Gateway"),
        ("/v1/base", 500, "internal_error", "Internal Server Error"),
    ],
)
async def test_platform_errors_map_to_their_own_status_and_code(
    client: httpx.AsyncClient,
    auth: dict[str, str],
    path: str,
    status: int,
    code: str,
    title: str,
) -> None:
    response = await client.get(path, headers=auth)

    assert response.status_code == status
    assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
    body = response.json()
    assert body["type"] == "about:blank"
    assert body["title"] == title
    assert body["status"] == status
    assert body["code"] == code
    assert body["instance"] == path
    assert body["request_id"] == response.headers["X-Request-ID"]


async def test_error_context_arrives_as_extension_members(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    body = (await client.get("/v1/conflict", headers=auth)).json()

    assert body["provider"] == "gemini"
    assert body["collection"] == "kb_gemini_v1"
    assert body["detail"] == "collection is not populated"


async def test_context_cannot_overwrite_the_standard_members(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get("/v1/context-cannot-shadow", headers=auth)
    body = response.json()

    assert response.status_code == 404
    assert body["status"] == 404
    assert body["code"] == "not_found"
    assert body["kept"] is True


async def test_an_unhandled_exception_becomes_a_500_that_leaks_nothing(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get("/v1/boom", headers=auth)

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "hunter2" not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


async def test_a_malformed_body_is_a_422_problem(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post("/v1/echo", json={"limit": "not-a-number"}, headers=auth)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["errors"][0]["loc"] == ["body", "limit"]


async def test_an_unknown_route_is_a_problem_too(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/nothing-here")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
    assert response.json()["code"] == "not_found"


async def test_a_wrong_method_keeps_the_allow_header(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post("/v1/not-found", headers=auth)

    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"
    assert "GET" in response.headers["allow"]
