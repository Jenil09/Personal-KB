"""The whole stack over ASGI: bearer auth, the mounted MCP app, and `/health`.

`mcp.Client` speaks to the server object directly and never runs the middleware,
so everything auth-related that a host actually meets — the `401`, the
`WWW-Authenticate` header, the auth context a tool body reads — only exists here.
This is also the only place that would catch the two ordering mistakes
`main.build_app` warns about: a `session_manager` touched before
`streamable_http_app()` raises at build time, and a lifespan that does not enter
`session_manager.run()` fails every request with `Task group is not initialized`.

The requests are raw JSON-RPC rather than an SDK client, because the SDK's own
HTTP client is `httpx2` and this suite's in-process transport is `httpx`. What is
being asserted is the wire behaviour anyway — status codes, headers, and the
session id — which is what a host sees and what a hand-written client would need.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette

from kb_client.testing import FakeService
from kb_mcp.config import KbMcpSettings
from kb_mcp.main import build_app

PROTOCOL = "2025-06-18"
JSON_AND_SSE = "application/json, text/event-stream"


@pytest.fixture
def app(settings: KbMcpSettings, service: FakeService) -> Starlette:
    return build_app(settings, transport=service.transport)


@asynccontextmanager
async def serving(app: Starlette) -> AsyncIterator[httpx.AsyncClient]:
    """A client over the app, with the lifespan entered.

    `ASGITransport` runs no lifespan events of its own, and without
    `session_manager.run()` having been entered every `/mcp` request fails at
    dispatch rather than at startup.

    A context manager rather than a fixture, deliberately. The session manager
    holds an anyio task group, and a cancel scope has to be exited by the task
    that entered it — a yielding fixture is torn down elsewhere, which turns
    every teardown into `Attempted to exit cancel scope in a different task`.
    """
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://kb-mcp.test"
        ) as client,
    ):
        yield client


def _initialize() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "test-host", "version": "1.0"},
        },
    }


def _headers(token: str | None = None, session: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": JSON_AND_SSE}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if session is not None:
        headers["mcp-session-id"] = session
    return headers


def _payload(response: httpx.Response) -> dict[str, Any]:
    """The JSON-RPC body, whether it arrived as JSON or as one SSE event."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data: "):
                frame: dict[str, Any] = json.loads(line.removeprefix("data: "))
                return frame
        raise AssertionError(f"no SSE data frame in {response.text!r}")
    body: dict[str, Any] = response.json()
    return body


async def _session(http: httpx.AsyncClient, token: str) -> str:
    """Initialize and return the session id, so a tool can be called on it."""
    response = await http.post("/mcp", json=_initialize(), headers=_headers(token))
    assert response.status_code == 200, response.text
    session: str = response.headers["mcp-session-id"]
    await http.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=_headers(token, session),
    )
    return session


async def _call_tool(
    http: httpx.AsyncClient,
    token: str,
    session: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    response = await http.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=_headers(token, session),
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = _payload(response)["result"]
    return result


# --- health ---------------------------------------------------------------


async def test_health_needs_no_token(app):
    """The container healthcheck runs without a credential, and reports only liveness.

    Whether `kb-api` is reachable is what the `kb_health` tool answers; a
    healthcheck that failed on an upstream outage would restart a process that is
    working.
    """
    async with serving(app) as http:
        response = await http.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- authentication -------------------------------------------------------


async def test_no_token_is_refused(app):
    async with serving(app) as http:
        response = await http.post("/mcp", json=_initialize(), headers=_headers())

    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["www-authenticate"]


async def test_an_unknown_token_is_refused_identically(app):
    """The two are indistinguishable, as they are in `kb-api`.

    One undifferentiated failure for every auth problem: a caller learns that
    their credential did not work, and not whether the key exists.
    """
    async with serving(app) as http:
        absent = await http.post("/mcp", json=_initialize(), headers=_headers())
        wrong = await http.post("/mcp", json=_initialize(), headers=_headers("not-a-real-token"))

    assert wrong.status_code == absent.status_code == 401
    assert wrong.headers["www-authenticate"] == absent.headers["www-authenticate"]


async def test_a_valid_token_opens_a_session(app):
    async with serving(app) as http:
        response = await http.post("/mcp", json=_initialize(), headers=_headers("host-token"))

    assert response.status_code == 200
    assert response.headers["mcp-session-id"]


async def test_a_foreign_host_header_is_not_a_421(app):
    """DNS-rebinding protection is off, deliberately, not by inheriting 0.0.0.0.

    The SDK 421s a Host that is not on its allowlist when the middleware is
    enabled. A production listener behind the tailnet must not do that: the
    Host `tailscale serve` forwards is not a stable name, and guessing one is
    how every real connection dies with 421. The request still has to
    authenticate — this is not an open door, it is the Host check staying out
    of the way.
    """
    headers = _headers("host-token")
    headers["Host"] = "evil.example"

    async with serving(app) as http:
        response = await http.post("/mcp", json=_initialize(), headers=headers)

    assert response.status_code == 200, response.text
    assert response.headers["mcp-session-id"]


async def test_a_near_miss_token_is_refused(app):
    """One character short of the real key. `hmac.compare_digest`, not `==`."""
    async with serving(app) as http:
        response = await http.post("/mcp", json=_initialize(), headers=_headers("host-toke"))

    assert response.status_code == 401


# --- scopes, through the middleware that sets the context -----------------


async def test_a_read_only_key_is_refused_the_ingest_tool(app):
    """The transport lets it in; the tool refuses it.

    `required_scopes` is empty on purpose — it is enforced endpoint-wide, so
    putting `search` there would `403` a write-only key on the whole transport
    and could never express this. The refusal has to come from the tool body, and
    it has to be a readable result rather than a protocol error.
    """
    async with serving(app) as http:
        session = await _session(http, "read-only-token")
        result = await _call_tool(
            http,
            "read-only-token",
            session,
            "kb_ingest_document",
            {"title": "Nope", "content": "Body.", "document_type": "note"},
        )

    assert result["isError"] is True
    assert "not allowed to write" in result["content"][0]["text"]


async def test_an_unrestricted_key_reaches_the_service(app, service: FakeService):
    async with serving(app) as http:
        session = await _session(http, "host-token")
        result = await _call_tool(
            http,
            "host-token",
            session,
            "kb_ingest_document",
            {"title": "From a host", "content": "Body text.", "document_type": "note"},
        )

    assert result.get("isError") is not True
    assert any(document["title"] == "From a host" for document in service.documents.values())


async def test_the_session_is_required_for_a_tool_call(app):
    """A token alone is not a session — the transport tracks state per connection."""
    async with serving(app) as http:
        response = await http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "kb_stats"},
            },
            headers=_headers("host-token"),
        )

    assert response.status_code == 400


# --- RFC 9728 metadata ----------------------------------------------------


async def test_the_metadata_route_is_absent_when_no_url_is_configured(app):
    """Unset `resource_server_url` suppresses the advertisement entirely.

    Right locally, where there is no external URL to name; a deployment sets it
    so that a host which *does* follow the advertisement fails somewhere
    diagnosable.
    """
    async with serving(app) as http:
        response = await http.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 404


async def test_the_metadata_is_published_when_a_url_is_configured(settings, service):
    """Note the path: no `/mcp` suffix.

    RFC 9728 inserts the well-known segment before the *resource path*, which is
    empty here — so a host looking under `/mcp/.well-known/...` finds nothing.
    """
    published = settings.model_copy(update={"resource_server_url": "https://kb.example.ts.net"})

    async with serving(build_app(published, transport=service.transport)) as http:
        response = await http.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 200
    body = response.json()
    assert body["resource"] == "https://kb.example.ts.net/"
    assert body["bearer_methods_supported"] == ["header"]
