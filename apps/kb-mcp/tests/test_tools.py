"""Every tool, driven in memory through `mcp.Client`, over the `/v1` contract fake.

`mcp.Client(server)` connects to the server object directly — no sockets, no
transport — so what these exercise is the tool surface a host sees: the schemas
advertised in `tools/list`, the arguments accepted, and the text returned. The
service underneath is `kb_client.testing.FakeService` over `httpx.MockTransport`,
which serves real problem+json and reproduces the cases that matter: a re-ingest
that embeds nothing (AD-008) and an ingest that supersedes by source (AD-020).

`get_access_token()` returns `None` under this client, because the SDK installs
`AuthContextMiddleware` in the ASGI stack and there is no ASGI here. The scope
checks are therefore exercised in `test_scopes.py` and `test_transport.py`, and
what remains in this file is everything else.
"""

from typing import Any
from uuid import UUID

import mcp
import pytest
from mcp.server import MCPServer
from mcp.types import CallToolResult

from kb_client.client import KbClient
from kb_mcp.config import KbMcpSettings
from kb_mcp.server import build_server


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


# --- the surface ----------------------------------------------------------


async def test_every_tool_is_advertised(server):
    async with mcp.Client(server) as connected:
        tools = {tool.name: tool for tool in (await connected.list_tools()).tools}

    assert set(tools) == {
        "kb_search",
        "kb_list_documents",
        "kb_get_document",
        "kb_stats",
        "kb_health",
        "kb_ingest_document",
    }
    search = tools["kb_search"].annotations
    ingest = tools["kb_ingest_document"].annotations
    assert search is not None and ingest is not None
    assert search.read_only_hint is True
    assert search.open_world_hint is False
    # Idempotent because AD-008's content hash and AD-020's supersede rule make
    # it so — a host may safely retry one.
    assert ingest.idempotent_hint is True
    assert ingest.read_only_hint is False


async def test_ingest_is_absent_when_it_is_not_allowed(settings, client):
    """Off means the tool does not exist, not that it exists and refuses."""
    read_only = settings.model_copy(update={"allow_ingest": False})

    async with mcp.Client(build_server(read_only, client)) as connected:
        names = {tool.name for tool in (await connected.list_tools()).tools}

    assert "kb_ingest_document" not in names
    assert "kb_search" in names


async def test_the_type_description_carries_the_vocabulary(server):
    async with mcp.Client(server) as connected:
        tools = {tool.name: tool for tool in (await connected.list_tools()).tools}

    described = tools["kb_ingest_document"].input_schema["properties"]["document_type"]
    assert "incident-report" in described["description"]
    assert "architecture" in described["description"]


# --- reads ----------------------------------------------------------------


async def test_search_returns_the_matching_passages(server, service):
    service.add("Chroma adapter", content="Chroma is the index, Postgres is the record.")

    result = await call(server, "kb_search", {"query": "where do vectors live"})

    body = text_of(result)
    assert result.is_error is not True
    assert "Chroma is the index" in body
    assert "Chroma adapter" in body


async def test_search_passes_its_filters_through(server, service):
    service.add("A note", type="note", tags=("chroma",))
    service.add("A runbook", type="sop", tags=("deploy",))

    await call(
        server,
        "kb_search",
        {"query": "anything", "document_type": "sop", "tags": ["deploy"], "match_all_tags": True},
    )

    sent = service.requests[-1]
    assert sent.url.path == "/v1/search"
    body = sent.read().decode()
    assert '"type":"sop"' in body.replace(" ", "")
    assert '"match_all_tags":true' in body.replace(" ", "")


async def test_search_says_so_when_nothing_matched(server):
    result = await call(server, "kb_search", {"query": "nothing is stored yet"})

    assert "No matches" in text_of(result)


async def test_listing_reports_titles_and_never_content(server, service):
    service.add("Visible title", content="SECRET BODY TEXT")

    body = text_of(await call(server, "kb_list_documents", {}))

    assert "Visible title" in body
    assert "SECRET BODY TEXT" not in body


async def test_listing_offers_the_next_page(server, service):
    for index in range(5):
        service.add(f"Document {index}")

    body = text_of(await call(server, "kb_list_documents", {"limit": 2}))

    assert "Showing 1-2 of 5" in body
    assert "offset=2" in body


async def test_get_document_returns_the_body(server, service):
    identifier = service.add("Design note", content="The whole body of the document.")

    body = text_of(await call(server, "kb_get_document", {"document_id": identifier}))

    assert "The whole body of the document." in body
    assert identifier in body


async def test_a_malformed_id_is_explained_rather_than_thrown(server):
    result = await call(server, "kb_get_document", {"document_id": "the third one"})

    assert result.is_error is True
    body = text_of(result)
    assert "is not a document id" in body
    assert "kb_search" in body
    # The complaint, not a traceback.
    assert "Traceback" not in body


async def test_stats_reports_the_corpus(server, service):
    service.add("One")

    body = text_of(await call(server, "kb_stats", {}))

    assert "1 documents" in body
    assert "text-embedding-3-small" in body


async def test_health_reports_the_service(server):
    assert "ok" in text_of(await call(server, "kb_health", {}))


# --- ingest ---------------------------------------------------------------


async def test_ingest_adds_a_document(server, service):
    result = await call(
        server,
        "kb_ingest_document",
        {"title": "New note", "content": "Something worth keeping.", "document_type": "note"},
    )

    body = text_of(result)
    assert result.is_error is not True
    assert "Ingested as" in body
    assert any(document["title"] == "New note" for document in service.documents.values())


async def test_re_ingesting_identical_content_embeds_nothing(server):
    arguments = {"title": "Same", "content": "Identical content.", "document_type": "note"}
    await call(server, "kb_ingest_document", arguments)

    body = text_of(await call(server, "kb_ingest_document", arguments))

    # AD-008: the content hash makes this a no-op rather than a duplicate, which
    # is what lets the tool be annotated idempotent.
    assert "Unchanged" in body
    assert "nothing was re-embedded" in body


async def test_ingesting_over_a_source_reports_what_it_replaced(server, service):
    original = service.add("Old version", source="ai-kb/design.md")

    body = text_of(
        await call(
            server,
            "kb_ingest_document",
            {
                "title": "New version",
                "content": "Rewritten.",
                "document_type": "architecture",
                "source": "ai-kb/design.md",
            },
        )
    )

    # AD-020: same source means supersede, and the model should be told rather
    # than left thinking it added a second copy.
    assert "Replaced 1 earlier document" in body
    assert original in body
    assert UUID(original) not in service.documents


async def test_ingest_refuses_empty_content(server):
    result = await call(
        server,
        "kb_ingest_document",
        {"title": "Titled", "content": "   ", "document_type": "note"},
    )

    assert result.is_error is True
    assert "content must not be empty" in text_of(result)


# --- defensive validation (3.14) ------------------------------------------


async def test_an_out_of_range_top_k_is_rejected(server):
    """Settles the contradiction in the v2 docs: schemas *are* enforced.

    The tools page says input schemas are checked before the handler runs and the
    migration guide says they are advertised but never applied. On 2.0.0 the
    former is true — this comes back as a validation error result. The body check
    stays anyway, for a client that is not the SDK.
    """
    result = await call(server, "kb_search", {"query": "anything", "top_k": 999})

    assert result.is_error is True
    assert "50" in text_of(result)


async def test_an_empty_query_is_rejected_in_the_body(server):
    """No schema can express this one, so the body is the only place it happens."""
    result = await call(server, "kb_search", {"query": "   "})

    assert result.is_error is True
    assert "query must not be empty" in text_of(result)


async def test_the_client_identifies_itself(server, service):
    """3.15 — MCP traffic is separable from CLI traffic in the audit trail."""
    await call(server, "kb_stats", {})

    assert service.requests[-1].headers["user-agent"].startswith("kb-mcp/")
