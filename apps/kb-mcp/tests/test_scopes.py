"""Per-tool scope checks (AD-024), asserted at the tool boundary.

The transport's own `required_scopes` is empty and has to be: the SDK enforces it
endpoint-wide in `RequireAuthMiddleware`, before a tool is chosen, so naming
`search` there would give a write-only key a `403` on the whole transport. These
checks in the tool bodies are therefore the only thing enforcing scopes at all —
load-bearing, not belt-and-braces.

The auth context is set here the way `AuthContextMiddleware` sets it, because
`mcp.Client` speaks to the server object directly and never runs the ASGI stack
that would. `test_transport.py` covers the same ground through a real request;
this file is where each tool is checked individually, which would be four
sessions and a lot of protocol noise over HTTP.
"""

from collections.abc import Iterator
from typing import Any

import mcp
import pytest
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.types import CallToolResult

from kb_client.client import KbClient
from kb_mcp.config import KbMcpSettings
from kb_mcp.server import build_server

READS = ["kb_search", "kb_list_documents", "kb_get_document", "kb_stats", "kb_health"]


@pytest.fixture
def server(settings: KbMcpSettings, client: KbClient) -> MCPServer:
    return build_server(settings, client)


@pytest.fixture
def as_reader() -> Iterator[None]:
    """A caller holding only `search`, as the middleware would present them."""
    reset = auth_context_var.set(
        AuthenticatedUser(AccessToken(token="opaque", client_id="reader", scopes=["search"]))
    )
    yield
    auth_context_var.reset(reset)


@pytest.fixture
def as_writer() -> Iterator[None]:
    """A caller holding only `write` — the other half of the split."""
    reset = auth_context_var.set(
        AuthenticatedUser(AccessToken(token="opaque", client_id="writer", scopes=["write"]))
    )
    yield
    auth_context_var.reset(reset)


async def call(
    server: MCPServer, name: str, arguments: dict[str, Any] | None = None
) -> CallToolResult:
    async with mcp.Client(server) as connected:
        return await connected.call_tool(name, arguments or {})


def text_of(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


async def test_a_read_only_key_may_not_ingest(server, as_reader):
    result = await call(
        server,
        "kb_ingest_document",
        {"title": "Nope", "content": "Body.", "document_type": "note"},
    )

    assert result.is_error is True
    assert "not allowed to write" in text_of(result)


async def test_a_read_only_key_may_search(server, service, as_reader):
    service.add("Something")

    result = await call(server, "kb_search", {"query": "something"})

    assert result.is_error is not True


async def test_a_write_key_may_ingest(server, as_writer):
    result = await call(
        server,
        "kb_ingest_document",
        {"title": "Yes", "content": "Body.", "document_type": "note"},
    )

    assert result.is_error is not True
    assert "Ingested as" in text_of(result)


@pytest.mark.parametrize("tool", READS)
async def test_every_read_tool_requires_search(server, service, tool, as_writer):
    """A key holding only `write` is refused every read, one tool at a time."""
    arguments = {
        "kb_search": {"query": "anything"},
        "kb_get_document": {"document_id": "00000000-0000-4000-8000-000000000000"},
    }.get(tool, {})

    result = await call(server, tool, arguments)

    assert result.is_error is True
    assert "not allowed to search" in text_of(result)


@pytest.fixture
def as_unrestricted() -> Iterator[None]:
    """A key named in no scope entry, which gets everything (AD-024)."""
    reset = auth_context_var.set(
        AuthenticatedUser(AccessToken(token="opaque", client_id="claude", scopes=[]))
    )
    yield
    auth_context_var.reset(reset)


async def test_an_unlisted_key_may_do_everything(server, as_unrestricted):
    """The permissive default, which `AccessToken.scopes` alone cannot express.

    It arrives as an empty list — there is nothing else it could be — and reading
    that as "no permissions" would invert the default that keeps a first deploy
    from failing on a valid credential.
    """
    read = await call(server, "kb_stats", {})
    write = await call(
        server,
        "kb_ingest_document",
        {"title": "Anything", "content": "Body.", "document_type": "note"},
    )

    assert read.is_error is not True
    assert write.is_error is not True
