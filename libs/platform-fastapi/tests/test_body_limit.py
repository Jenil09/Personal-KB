"""The application-layer body cap (Technical Design §8).

Two paths, and the second is the one worth having. A client that declares
`Content-Length` is refused before a byte of body is read. A client that does
not declare one — chunked transfer — has to be counted as it arrives, and it is
exactly the caller who declines to state a size that a header-only check would
let through.
"""

from collections.abc import AsyncIterator
from http import HTTPStatus

import httpx
import pytest
from fastapi import APIRouter
from pydantic import SecretStr

from platform_fastapi import BodySizeLimitMiddleware, HttpServiceSettings, create_app

LIMIT = 1024

SETTINGS = HttpServiceSettings(
    api_keys={"test": SecretStr("secret")},
    max_body_bytes=LIMIT,
)

AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture
def client() -> httpx.AsyncClient:
    router = APIRouter()

    @router.post("/echo")
    async def echo(body: dict[str, str]) -> dict[str, int]:
        return {"length": len(body.get("payload", ""))}

    app = create_app(SETTINGS, routers=(router,))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_a_body_inside_the_limit_is_served(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/echo", json={"payload": "x" * 100}, headers=AUTH)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"length": 100}


async def test_a_declared_oversize_body_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/echo", json={"payload": "x" * (LIMIT * 2)}, headers=AUTH)

    assert response.status_code == HTTPStatus.CONTENT_TOO_LARGE
    assert response.json()["code"] == "payload_too_large"


async def test_the_refusal_is_problem_json_carrying_the_limit(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/v1/echo", json={"payload": "x" * (LIMIT * 2)}, headers=AUTH)

    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["limit_bytes"] == LIMIT


async def test_an_undeclared_oversize_body_is_refused_too(client: httpx.AsyncClient) -> None:
    # No `Content-Length`: httpx streams a generator body chunked. The header
    # check cannot see this one, so the running byte count has to.
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"x" * 512

    response = await client.post("/v1/echo", content=chunks(), headers=AUTH)

    assert response.status_code == HTTPStatus.CONTENT_TOO_LARGE


async def test_the_cap_applies_before_authentication(client: httpx.AsyncClient) -> None:
    # The middleware sits outside the router, so an unauthenticated flood is
    # refused on size rather than being buffered first and rejected after.
    response = await client.post("/v1/echo", json={"payload": "x" * (LIMIT * 2)})

    assert response.status_code == HTTPStatus.CONTENT_TOO_LARGE


async def test_health_is_never_capped(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == HTTPStatus.OK


def test_a_nonsense_limit_is_rejected_at_construction() -> None:
    # `HttpServiceSettings` already refuses a non-positive value, so this path
    # is only reachable by wiring the middleware directly. It is still worth
    # guarding: the class is public and a service could add it by hand.
    with pytest.raises(ValueError, match="max_bytes"):
        BodySizeLimitMiddleware(create_app(SETTINGS), max_bytes=0)
