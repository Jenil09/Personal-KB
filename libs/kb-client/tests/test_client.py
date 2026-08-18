"""The client against the `/v1` contract.

Two groups. The first is what goes out — the bearer header, a UUID request id,
the provider omitted rather than nulled. The second is what comes back, and in
particular that a problem+json body becomes the same `PlatformError` subclass
the service raised, because every `except NotFoundError` in this package depends
on it.
"""

import json
from uuid import UUID, uuid4

import pytest

from kb_client.client import KbClient
from kb_client.settings import KbClientSettings
from platform_core import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    PlatformError,
    RateLimitedError,
    UpstreamError,
    ValidationError,
)


@pytest.fixture
def client(service, settings: KbClientSettings) -> KbClient:
    return KbClient(settings, transport=service.transport)


# --- the request ----------------------------------------------------------


async def test_the_api_key_is_sent_as_a_bearer_token(
    client: KbClient, service, api_key: str
) -> None:
    await client.list_documents()

    assert service.requests[-1].headers["Authorization"] == f"Bearer {api_key}"


async def test_every_request_carries_a_uuid_request_id(client: KbClient, service) -> None:
    """A non-UUID is replaced by the service and never reaches the audit row."""
    await client.list_documents()

    UUID(service.requests[-1].headers["X-Request-ID"])


async def test_request_ids_are_not_reused(client: KbClient, service) -> None:
    await client.list_documents()
    await client.list_documents()

    first, second = (request.headers["X-Request-ID"] for request in service.requests[-2:])
    assert first != second


async def test_the_base_url_gets_the_v1_prefix(client: KbClient, service) -> None:
    await client.list_documents()

    assert service.requests[-1].url.path == "/v1/documents"


# --- documents ------------------------------------------------------------


async def test_list_documents_parses_a_page(client: KbClient, service) -> None:
    service.add("One")
    service.add("Two")

    page = await client.list_documents()

    assert page.total == 2
    assert {document.title for document in page.documents} == {"One", "Two"}


async def test_list_documents_sends_only_the_filters_that_were_set(
    client: KbClient, service
) -> None:
    await client.list_documents(document_type="architecture")

    params = service.requests[-1].url.params
    assert params["type"] == "architecture"
    assert "source" not in params
    # `match_all_tags` rides with `tags`; sending it alone says nothing.
    assert "match_all_tags" not in params


async def test_list_documents_forwards_tag_filters(client: KbClient, service) -> None:
    service.add("Tagged", tags=("ansible", "hardening"))
    service.add("Other", tags=("unrelated",))

    page = await client.list_documents(tags=["ansible"])

    assert [document.title for document in page.documents] == ["Tagged"]


async def test_match_all_tags_requires_every_tag(client: KbClient, service) -> None:
    service.add("Both", tags=("ansible", "hardening"))
    service.add("One", tags=("ansible",))

    page = await client.list_documents(tags=["ansible", "hardening"], match_all_tags=True)

    assert [document.title for document in page.documents] == ["Both"]


async def test_get_document_returns_the_content_and_provenance(client: KbClient, service) -> None:
    identifier = service.add("Readable", content="# Readable\n\nthe body")

    detail = await client.get_document(UUID(identifier))

    assert detail.content == "# Readable\n\nthe body"
    assert detail.ingested_by_key_id == "cli"


async def test_get_document_raises_not_found_for_an_unknown_id(client: KbClient) -> None:
    with pytest.raises(NotFoundError):
        await client.get_document(uuid4())


async def test_delete_reports_that_it_removed_something(client: KbClient, service) -> None:
    identifier = service.add("Doomed")

    assert await client.delete_document(UUID(identifier)) is True
    assert identifier not in service.documents


async def test_deleting_twice_succeeds_and_says_nothing_was_there(
    client: KbClient, service
) -> None:
    """The endpoint is idempotent (PRD §6.6); the header is the only difference."""
    identifier = service.add("Doomed")
    await client.delete_document(UUID(identifier))

    assert await client.delete_document(UUID(identifier)) is False


# --- pagination -----------------------------------------------------------


async def test_iter_documents_walks_every_page(client: KbClient, service) -> None:
    for index in range(25):
        service.add(f"Doc {index}")

    collected = [
        document
        async for page in client.iter_documents(page_size=10)
        for document in page.documents
    ]

    assert len(collected) == 25
    assert len({document.id for document in collected}) == 25


async def test_iter_documents_stops_when_the_last_page_is_exactly_full(
    client: KbClient, service
) -> None:
    """A full final page is normal; treating it as "more" never terminates."""
    for index in range(20):
        service.add(f"Doc {index}")

    pages = [page async for page in client.iter_documents(page_size=10)]

    assert len(pages) == 2


async def test_iter_documents_handles_an_empty_corpus(client: KbClient) -> None:
    pages = [page async for page in client.iter_documents()]

    assert len(pages) == 1
    assert pages[0].documents == ()


# --- ingest ---------------------------------------------------------------


async def test_ingest_sends_the_document(client: KbClient, service) -> None:
    result = await client.ingest(
        title="New", content="# New\n\nbody", document_type="note", tags=["a"]
    )

    assert result.chunks_created > 0
    assert result.unchanged is False
    body = json.loads(service.requests[-1].content)
    assert body["title"] == "New"
    assert body["tags"] == ["a"]


async def test_ingest_omits_the_provider_when_none_is_configured(client: KbClient, service) -> None:
    """An absent field lets the service apply its own default (AD-006)."""
    await client.ingest(title="New", content="body", document_type="note")

    assert "provider" not in json.loads(service.requests[-1].content)


async def test_ingest_uses_the_configured_provider(service, settings: KbClientSettings) -> None:
    client = KbClient(
        settings.model_copy(update={"provider": "gemini"}), transport=service.transport
    )

    await client.ingest(title="New", content="body", document_type="note")

    assert json.loads(service.requests[-1].content)["provider"] == "gemini"


async def test_an_explicit_provider_beats_the_configured_one(
    service, settings: KbClientSettings
) -> None:
    client = KbClient(
        settings.model_copy(update={"provider": "gemini"}), transport=service.transport
    )

    await client.ingest(title="New", content="body", document_type="note", provider="openai")

    assert json.loads(service.requests[-1].content)["provider"] == "openai"


async def test_re_ingesting_identical_content_embeds_nothing(client: KbClient, service) -> None:
    """AD-008's re-ingest property, as the CLI reports it."""
    await client.ingest(title="Same", content="the body", document_type="note")

    second = await client.ingest(title="Same", content="the body", document_type="note")

    assert second.unchanged is True
    assert second.chunks_created == 0


async def test_ingesting_over_a_source_reports_what_it_replaced(client: KbClient, service) -> None:
    """AD-020 supersedes on `source`; silence here would hide a replacement."""
    first = await client.ingest(
        title="V1", content="first", document_type="note", source="notes.md"
    )

    second = await client.ingest(
        title="V2", content="second", document_type="note", source="notes.md"
    )

    assert second.superseded == (first.document_id,)


# --- search and stats -----------------------------------------------------


async def test_search_returns_ranked_hits(client: KbClient, service) -> None:
    service.add("Architecture", content="bare metal kubernetes")

    response = await client.search("kubernetes")

    assert response.results[0].metadata.title == "Architecture"
    assert response.results[0].score > 0


async def test_search_sends_only_the_filters_that_were_set(client: KbClient, service) -> None:
    await client.search("anything", document_type="architecture")

    filters = json.loads(service.requests[-1].content)["filters"]
    assert filters == {"type": "architecture"}


async def test_search_respects_top_k(client: KbClient, service) -> None:
    for index in range(5):
        service.add(f"Doc {index}")

    response = await client.search("anything", top_k=2)

    assert len(response.results) == 2


async def test_stats_parses_the_admin_response(client: KbClient, service) -> None:
    service.add("One")

    stats = await client.stats()

    assert stats.total_documents == 1
    assert stats.collections[0].provider == "openai"
    assert stats.degraded is False


async def test_health_needs_no_key(client: KbClient) -> None:
    assert (await client.health())["status"] == "ok"


# --- errors ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, "unauthenticated", AuthenticationError),
        (403, "insufficient_scope", AuthorizationError),
        (404, "not_found", NotFoundError),
        (409, "conflict", ConflictError),
        (422, "validation_error", ValidationError),
        (429, "rate_limited", RateLimitedError),
        (502, "upstream_error", UpstreamError),
    ],
)
async def test_a_problem_body_becomes_its_own_error_class(
    client: KbClient, service, status: int, code: str, expected: type[PlatformError]
) -> None:
    service.fail_next = (status, code, "the service said so")

    with pytest.raises(expected) as caught:
        await client.list_documents()

    # The service's own sentence, not one this client invented.
    assert "the service said so" in str(caught.value)


async def test_the_request_id_survives_onto_the_error(client: KbClient, service) -> None:
    """It is the handle for finding the tier-1 audit row (AD-013)."""
    service.fail_next = (502, "upstream_error", "chroma is down")

    with pytest.raises(UpstreamError) as caught:
        await client.list_documents()

    UUID(caught.value.context["request_id"])


async def test_a_wrong_key_is_an_authentication_error(
    service, settings: KbClientSettings, api_key: str
) -> None:
    client = KbClient(
        settings.model_copy(update={"api_key": settings.api_key.__class__("wrong")}),
        transport=service.transport,
    )

    with pytest.raises(AuthenticationError):
        await client.list_documents()


async def test_a_non_problem_body_still_produces_the_right_class(client: KbClient, service) -> None:
    """An HTML error page from something in front of the service is still a 502."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    broken = KbClient(client._settings, transport=httpx.MockTransport(handler))

    with pytest.raises(UpstreamError):
        await broken.list_documents()


async def test_an_unreachable_service_is_an_upstream_error(settings: KbClientSettings) -> None:
    """The common laptop case: a dropped tailnet session, or `just up` not run."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = KbClient(settings, transport=httpx.MockTransport(handler))

    with pytest.raises(UpstreamError, match="Could not reach"):
        await client.list_documents()


async def test_a_timeout_is_an_upstream_error(settings: KbClientSettings) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    client = KbClient(settings, transport=httpx.MockTransport(handler))

    with pytest.raises(UpstreamError, match="did not answer in time"):
        await client.list_documents()


async def test_the_client_closes_its_connection_pool(service, settings: KbClientSettings) -> None:
    async with KbClient(settings, transport=service.transport) as client:
        await client.list_documents()

    assert client._client.is_closed
