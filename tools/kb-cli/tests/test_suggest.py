"""The metadata suggester, driven against a fake `/chat/completions` (AD-026).

No network and no provider SDK — the same `MockTransport` seam `test_client.py`
uses. What is asserted is the two halves that actually decide whether this
feature is usable: that a wrong answer is corrected rather than believed, and
that a local runtime which refuses `json_schema` or wraps its JSON in prose is
still understood.
"""

import json

import httpx
import pytest

from kb_cli.config import SuggestSettings
from kb_cli.suggest import collect_tag_vocabulary, suggest_metadata
from platform_core import ConfigurationError, UpstreamError

DOCUMENT = "# Redshift Architecture\n\nBare metal Kubernetes with Prometheus.\n"


async def suggest(model, settings: SuggestSettings, content: str = DOCUMENT, **kwargs):
    async with model.client() as client:
        return await suggest_metadata(content, settings=settings, client=client, **kwargs)


# --- the happy path -------------------------------------------------------


async def test_a_suggestion_is_parsed(model, suggest_settings: SuggestSettings) -> None:
    suggestion = await suggest(model, suggest_settings)

    assert suggestion.title == "Redshift Architecture"
    assert suggestion.type == "architecture"
    assert suggestion.tags == ("kubernetes", "observability")
    assert suggestion.model == "gpt-4o-mini"
    assert suggestion.truncated is False
    assert suggestion.coerced_type is None


async def test_the_request_is_deterministic_and_asks_for_json(
    model, suggest_settings: SuggestSettings
) -> None:
    await suggest(model, suggest_settings)

    sent = model.requests[0]
    assert sent["model"] == "gpt-4o-mini"
    # A suggester that answers differently on two presses is one you stop
    # trusting; it is also what would make every assertion here flaky.
    assert sent["temperature"] == 0
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"]["properties"]["type"]["enum"] == [
        "profile",
        "architecture",
        "incident-report",
        "sop",
        "philosophy",
        "migration",
        "blog",
        "note",
    ]


async def test_the_known_tags_reach_the_prompt(model, suggest_settings: SuggestSettings) -> None:
    await suggest(model, suggest_settings, known_tags=("ansible", "wazuh"))

    system = model.requests[0]["messages"][0]["content"]
    assert "ansible, wazuh" in system


async def test_no_key_is_sent_when_none_is_configured(model) -> None:
    """LM Studio needs no credential, and inventing one would be sent to it."""
    captured: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers)
        return model.handle(request)

    settings = SuggestSettings(base_url="http://model.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await suggest_metadata(DOCUMENT, settings=settings, client=client)

    assert "authorization" not in captured[0]


async def test_a_configured_key_is_sent_as_a_bearer_token(model) -> None:
    captured: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers)
        return model.handle(request)

    settings = SuggestSettings(base_url="http://model.test/v1", api_key="sk-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await suggest_metadata(DOCUMENT, settings=settings, client=client)

    assert captured[0]["authorization"] == "Bearer sk-test"


# --- the answer is never simply believed ----------------------------------


async def test_a_type_outside_the_list_becomes_note_and_says_so(
    model, suggest_settings: SuggestSettings
) -> None:
    model.answer = {"title": "Rebuilding The Cluster", "type": "runbook", "tags": ["k8s"]}

    suggestion = await suggest(model, suggest_settings)

    assert suggestion.type == "note"
    assert suggestion.coerced_type == "runbook"
    assert "type coerced from 'runbook'" in suggestion.notes


async def test_a_correct_note_is_not_reported_as_a_correction(
    model, suggest_settings: SuggestSettings
) -> None:
    model.answer = {"title": "A Thought", "type": "note", "tags": []}

    suggestion = await suggest(model, suggest_settings)

    assert suggestion.type == "note"
    assert suggestion.coerced_type is None


async def test_tags_are_normalised_and_capped(model, suggest_settings: SuggestSettings) -> None:
    model.answer = {
        "title": "Everything",
        "type": "note",
        "tags": ["Ansible", "ansible", "Wazuh SIEM", "a", "b", "c", "d", "e", "f"],
    }

    suggestion = await suggest(model, suggest_settings)

    assert suggestion.tags == ("ansible", "wazuh-siem", "a", "b", "c", "d")


async def test_a_missing_title_falls_back_rather_than_failing(
    model, suggest_settings: SuggestSettings
) -> None:
    model.answer = {"title": "", "type": "note", "tags": []}

    suggestion = await suggest(model, suggest_settings, fallback_title="From The Heading")

    assert suggestion.title == "From The Heading"


async def test_a_missing_title_with_no_fallback_is_an_upstream_error(
    model, suggest_settings: SuggestSettings
) -> None:
    model.answer = {"title": "   ", "type": "note", "tags": []}

    with pytest.raises(UpstreamError):
        await suggest(model, suggest_settings)


async def test_tags_that_are_not_a_list_are_dropped(
    model, suggest_settings: SuggestSettings
) -> None:
    model.answer = {"title": "Something", "type": "note", "tags": "ansible, wazuh"}

    suggestion = await suggest(model, suggest_settings)

    assert suggestion.tags == ()


# --- what a local model actually does -------------------------------------


async def test_a_rejected_json_schema_is_retried_as_a_json_object(
    model, suggest_settings: SuggestSettings
) -> None:
    """llama.cpp-backed runtimes answer 400 for a `response_format` they lack."""
    model.reject_json_schema = True

    suggestion = await suggest(model, suggest_settings)

    assert suggestion.title == "Redshift Architecture"
    assert [request["response_format"]["type"] for request in model.requests] == [
        "json_schema",
        "json_object",
    ]


async def test_json_in_a_fenced_block_is_understood(
    model, suggest_settings: SuggestSettings
) -> None:
    model.reply = f"```json\n{json.dumps(model.answer)}\n```"

    assert (await suggest(model, suggest_settings)).type == "architecture"


async def test_json_after_a_sentence_of_preamble_is_understood(
    model, suggest_settings: SuggestSettings
) -> None:
    model.reply = f"Sure! Here is the metadata:\n\n{json.dumps(model.answer)}\n\nHope that helps."

    assert (await suggest(model, suggest_settings)).title == "Redshift Architecture"


async def test_nested_braces_do_not_truncate_the_object(
    model, suggest_settings: SuggestSettings
) -> None:
    model.reply = json.dumps(
        {"title": "A {brace} in the title", "type": "note", "tags": ["x"], "why": {"a": "b"}}
    )

    assert (await suggest(model, suggest_settings)).title == "A {brace} in the title"


async def test_prose_with_no_json_at_all_is_an_upstream_error(
    model, suggest_settings: SuggestSettings
) -> None:
    model.reply = "I am afraid I cannot classify this document."

    with pytest.raises(UpstreamError, match="did not return JSON"):
        await suggest(model, suggest_settings)


# --- failure --------------------------------------------------------------


async def test_an_unconfigured_endpoint_names_the_setting_that_fixes_it() -> None:
    with pytest.raises(ConfigurationError, match=r"suggest\.base_url"):
        await suggest_metadata(DOCUMENT, settings=SuggestSettings())


async def test_a_refused_connection_is_configuration_not_upstream(
    suggest_settings: SuggestSettings,
) -> None:
    """The model server is not running, which is a local fact with a local fix."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(ConfigurationError, match="Is the model server running"):
            await suggest_metadata(DOCUMENT, settings=suggest_settings, client=client)


async def test_a_server_error_carries_the_providers_own_sentence(
    model, suggest_settings: SuggestSettings
) -> None:
    model.status = 500
    model.error = {"error": {"message": "model is loading"}}

    with pytest.raises(UpstreamError, match="model is loading"):
        await suggest(model, suggest_settings)


async def test_an_unauthorised_key_is_reported(model, suggest_settings: SuggestSettings) -> None:
    model.status = 401
    model.error = {"error": {"message": "Incorrect API key provided"}}

    with pytest.raises(UpstreamError, match="Incorrect API key"):
        await suggest(model, suggest_settings)


async def test_an_empty_document_is_refused_before_the_call(
    model, suggest_settings: SuggestSettings
) -> None:
    with pytest.raises(ConfigurationError):
        await suggest(model, suggest_settings, content="   \n\n  ")
    assert model.requests == []


# --- long documents -------------------------------------------------------


def long_document() -> str:
    sections = [f"## Section {index}\n\n{'body text. ' * 40}\n" for index in range(30)]
    return "# Long Document\n\n" + "\n".join(sections) + "\n## Final Section\n\nThe last words.\n"


async def test_a_short_document_is_sent_whole(model, suggest_settings: SuggestSettings) -> None:
    suggestion = await suggest(model, suggest_settings)

    assert suggestion.truncated is False
    assert model.requests[0]["messages"][1]["content"] == DOCUMENT


async def test_a_long_document_is_reduced_and_flagged(
    model, suggest_settings: SuggestSettings
) -> None:
    content = long_document()
    assert len(content) > suggest_settings.max_content_chars

    suggestion = await suggest(model, suggest_settings, content=content)

    sent = model.requests[0]["messages"][1]["content"]
    assert suggestion.truncated is True
    assert len(sent) < len(content)
    assert "truncated" in sent


async def test_reduction_keeps_the_headings_from_the_elided_middle(
    model, suggest_settings: SuggestSettings
) -> None:
    """The outline is the highest signal per character for both title and type."""
    await suggest(model, suggest_settings, content=long_document())

    sent = model.requests[0]["messages"][1]["content"]
    assert "OUTLINE:" in sent
    assert "## Section 15" in sent
    assert "## Final Section" in sent


async def test_reduction_is_deterministic(model, suggest_settings: SuggestSettings) -> None:
    """AD-008's rule, and the reason every assertion above can be exact."""
    content = long_document()

    await suggest(model, suggest_settings, content=content)
    await suggest(model, suggest_settings, content=content)

    assert model.requests[0]["messages"] == model.requests[1]["messages"]


# --- the tag vocabulary ---------------------------------------------------


async def test_the_vocabulary_is_the_corpus_tags_most_common_first(service, settings) -> None:
    from kb_cli.client import KbClient

    service.add("One", tags=("ansible", "security"))
    service.add("Two", tags=("security",))
    service.add("Three", tags=("security", "wazuh"))

    client = KbClient(settings, transport=service.transport)
    try:
        assert await collect_tag_vocabulary(client) == ("security", "ansible", "wazuh")
    finally:
        await client.aclose()


async def test_the_vocabulary_is_empty_for_an_empty_corpus(service, settings) -> None:
    from kb_cli.client import KbClient

    client = KbClient(settings, transport=service.transport)
    try:
        assert await collect_tag_vocabulary(client) == ()
    finally:
        await client.aclose()
