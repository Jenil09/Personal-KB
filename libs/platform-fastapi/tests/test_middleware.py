"""Correlation ID in and out, and the access log."""

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.types import Message, Scope

from platform_core import NotFoundError, get_request_id
from platform_fastapi import RequestContextMiddleware


class LogRecorder:
    """Stands in for the module logger so log lines can be asserted on."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str):
        def log(event: str, **fields: Any) -> None:
            self.events.append((level, event, fields))

        return log

    def __getattr__(self, level: str):
        return self._record(level)

    def of(self, event: str) -> list[tuple[str, str, dict[str, Any]]]:
        return [entry for entry in self.events if entry[1] == event]


@pytest.fixture
def router() -> APIRouter:
    api = APIRouter()

    @api.get("/stream")
    async def stream() -> StreamingResponse:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"partial"
            raise RuntimeError("failed mid-stream")

        return StreamingResponse(chunks())

    @api.get("/echo-context")
    async def echo_context() -> dict[str, str | None]:
        return {"bound_request_id": get_request_id()}

    @api.get("/missing")
    async def missing() -> None:
        raise NotFoundError("nope")

    @api.get("/boom")
    async def boom() -> None:
        raise RuntimeError("unhandled")

    return api


@pytest.fixture
async def client(make_app, client_for, router):
    async with client_for(make_app(routers=[router])) as http_client:
        yield http_client


@pytest.fixture
def auth(valid_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {valid_key}"}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> LogRecorder:
    recorder = LogRecorder()
    monkeypatch.setattr("platform_fastapi.middleware._logger", recorder)
    return recorder


async def test_a_request_id_is_generated_when_none_is_supplied(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/health")

    assert UUID(response.headers["X-Request-ID"])


async def test_a_supplied_uuid_is_propagated(client: httpx.AsyncClient) -> None:
    supplied = "3f8c8d4e-2a4b-4c9a-9a44-6b1d0f8f2b11"
    response = await client.get("/health", headers={"X-Request-ID": supplied})

    assert response.headers["X-Request-ID"] == supplied


async def test_a_non_uuid_request_id_is_replaced_not_propagated(
    client: httpx.AsyncClient,
) -> None:
    """The audit column is a Postgres UUID — an unparseable header must not reach it."""
    response = await client.get("/health", headers={"X-Request-ID": "'; DROP TABLE --"})

    assert response.headers["X-Request-ID"] != "'; DROP TABLE --"
    assert UUID(response.headers["X-Request-ID"])


async def test_the_id_in_the_header_is_the_one_bound_for_logging(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get("/v1/echo-context", headers=auth)

    assert response.json()["bound_request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize("path", ["/health", "/v1/missing", "/v1/boom", "/v1/nothing-here"])
async def test_every_response_carries_the_header(
    client: httpx.AsyncClient, auth: dict[str, str], path: str
) -> None:
    response = await client.get(path, headers=auth)

    assert UUID(response.headers["X-Request-ID"])


async def test_one_access_log_line_per_request_with_the_caller_and_duration(
    client: httpx.AsyncClient, auth: dict[str, str], recorder: LogRecorder
) -> None:
    await client.get("/v1/echo-context", headers=auth)

    (level, _, fields) = recorder.of("request")[0]
    assert len(recorder.of("request")) == 1
    assert level == "info"
    assert fields["method"] == "GET"
    assert fields["path"] == "/v1/echo-context"
    assert fields["status"] == 200
    assert fields["key_id"] == "n8n"
    assert fields["duration_ms"] >= 0


async def test_an_unauthenticated_request_is_logged_without_a_key_id(
    client: httpx.AsyncClient, recorder: LogRecorder
) -> None:
    await client.get("/v1/echo-context")

    (level, _, fields) = recorder.of("request")[0]
    assert level == "warning"
    assert fields["status"] == 401
    assert fields["key_id"] is None


async def test_a_failed_request_is_logged_at_error_with_the_exception(
    client: httpx.AsyncClient, auth: dict[str, str], recorder: LogRecorder
) -> None:
    await client.get("/v1/boom", headers=auth)

    assert recorder.of("request")[0][0] == "error"
    assert recorder.of("request")[0][2]["status"] == 500
    assert isinstance(recorder.of("request_unhandled_error")[0][2]["exc_info"], RuntimeError)


async def test_a_failure_after_the_headers_are_sent_does_not_send_a_second_response(
    client: httpx.AsyncClient, auth: dict[str, str], recorder: LogRecorder
) -> None:
    """The 200 is already on the wire; contradicting it with a 500 would be a
    protocol error, so the failure is logged and the connection is torn down."""
    response = await client.get("/v1/stream", headers=auth)

    assert response.status_code == 200
    assert recorder.of("request_unhandled_error")
    assert recorder.of("request")[0][2]["status"] == 200


async def test_non_http_traffic_passes_straight_through() -> None:
    seen: list[Scope] = []

    async def downstream(scope: Scope, receive: object, send: object) -> None:
        seen.append(scope)

    async def receive() -> Message:  # pragma: no cover - never called
        return {"type": "websocket.connect"}

    async def send(message: Message) -> None:  # pragma: no cover - never called
        return None

    scope: Scope = {"type": "websocket", "path": "/ws", "headers": []}
    await RequestContextMiddleware(downstream)(scope, receive, send)

    assert seen == [scope]
    assert "state" not in scope


async def test_the_context_does_not_leak_between_requests(client: httpx.AsyncClient) -> None:
    first = await client.get("/health")
    second = await client.get("/health")

    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    assert get_request_id() is None
