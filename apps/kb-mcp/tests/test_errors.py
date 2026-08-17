"""Every `PlatformError` reaches the model as something it can read and act on.

`KbClient` has already reversed problem+json into the error hierarchy; what is
asserted here is the second half — that each one becomes a `CallToolResult` with
`is_error`, never an `MCPError`. The distinction matters because an `MCPError` is
a JSON-RPC protocol failure: the host sees a broken server and the model sees
nothing to retry against.

The other half is what must *not* appear. `UpstreamError`'s own message embeds
the service's base URL, because `KbClient` builds it that way for an operator at
a terminal; a model is not an operator, and where this server is, is not its
business.
"""

from typing import Any

import httpx
import mcp
import pytest
from mcp.server import MCPServer
from mcp.types import CallToolResult

from kb_client.client import KbClient
from kb_mcp.config import KbMcpSettings
from kb_mcp.server import build_server

# (status, problem+json code) → the failure the service would report.
FAILURES = [
    (401, "unauthenticated"),
    (403, "forbidden"),
    (404, "not_found"),
    (409, "conflict"),
    (413, "payload_too_large"),
    (422, "validation_failed"),
    (429, "rate_limited"),
    (502, "upstream_error"),
]


@pytest.fixture
def server(settings: KbMcpSettings, client: KbClient) -> MCPServer:
    return build_server(settings, client)


async def call(
    server: MCPServer, name: str, arguments: dict[str, Any] | None = None
) -> CallToolResult:
    async with mcp.Client(server) as connected:
        return await connected.call_tool(name, arguments or {})


def text_of(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


@pytest.mark.parametrize(("status", "code"), FAILURES)
async def test_every_service_failure_is_a_readable_result(server, service, status, code):
    service.fail_next = (status, code, "The service explained what went wrong.")

    result = await call(server, "kb_search", {"query": "anything"})

    # A result the model reads, not a protocol error it never sees.
    assert result.is_error is True
    body = text_of(result)
    assert body.strip()
    assert "Traceback" not in body


async def test_a_missing_document_says_so_in_words(server, service):
    service.fail_next = (404, "not_found", "No document with id 123.")

    body = text_of(await call(server, "kb_stats", {}))

    assert "No document with that id" in body
    # And says how to get a current one.
    assert "kb_search" in body


async def test_an_unreachable_service_does_not_leak_where_it_is(settings):
    """`UpstreamError` is the one message that is rewritten rather than passed through.

    `KbClient` builds it from the base URL — right for an operator reading a
    terminal, wrong for a model, which should not learn the service's address
    from a failed call.
    """

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    unreachable = KbClient(settings.kb_api, transport=httpx.MockTransport(refuse))
    try:
        body = text_of(await call(build_server(settings, unreachable), "kb_stats", {}))
    finally:
        await unreachable.aclose()

    assert "did not answer" in body
    assert settings.kb_api.base_url not in body
    assert "kb.test" not in body


async def test_a_rejected_credential_is_not_presented_as_retryable(server, service):
    service.fail_next = (401, "unauthenticated", "API key is not recognised.")

    body = text_of(await call(server, "kb_stats", {}))

    assert "configuration problem" in body
    assert "retrying will not fix it" in body


async def test_the_api_key_never_appears_in_a_failure(server, service, settings):
    service.fail_next = (403, "forbidden", "This key may not write.")

    body = text_of(await call(server, "kb_search", {"query": "x"}))

    assert settings.kb_api.api_key.get_secret_value() not in body


async def test_a_rate_limit_tells_the_model_to_wait(server, service):
    service.fail_next = (429, "rate_limited", "Too many requests.")

    body = text_of(await call(server, "kb_search", {"query": "x"}))

    assert "Wait before retrying" in body
