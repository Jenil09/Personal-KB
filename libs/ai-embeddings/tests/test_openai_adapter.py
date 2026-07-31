"""The OpenAI driver against the shared contract, plus what is specific to it.

The stub sits at the HTTP transport rather than in place of `AsyncOpenAI`, so
the SDK's own request building, response parsing, and error types are all in the
path. A test that mocks the client tests the mock.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from ai_embeddings import EmbeddingProviderSettings, TokenSource
from ai_embeddings.adapters.openai import TEXT_EMBEDDING_3_SMALL, OpenAIEmbeddingProvider
from ai_embeddings.testing import EmbeddingProviderContract, basis_vector
from ai_embeddings.tokens import count_tokens
from platform_core import UpstreamError

DIMENSIONS = TEXT_EMBEDDING_3_SMALL.dimensions


def build(
    recorded: list[httpx.Request],
    *,
    status: int = 200,
    vectors: int | None = None,
    width: int = DIMENSIONS,
    unreachable: bool = False,
    reverse: bool = True,
) -> OpenAIEmbeddingProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if unreachable:
            raise httpx.ConnectError("no route to host", request=request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "nope", "type": "x"}})

        inputs = json.loads(request.content)["input"]
        count = len(inputs) if vectors is None else vectors
        data = [
            {"object": "embedding", "index": index, "embedding": basis_vector(index, width)}
            for index in range(count)
        ]
        # Returned out of order on purpose: the API documents order as
        # unspecified and carries an `index`, so the driver has to sort.
        if reverse:
            data.reverse()
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": TEXT_EMBEDDING_3_SMALL.model_id,
                "usage": {
                    "prompt_tokens": sum(count_tokens(text) for text in inputs),
                    "total_tokens": sum(count_tokens(text) for text in inputs),
                },
            },
        )

    return OpenAIEmbeddingProvider(
        EmbeddingProviderSettings(api_key="sk-test", max_retries=0),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def make_provider(requests: list[httpx.Request]) -> Callable[..., OpenAIEmbeddingProvider]:
    def factory(**overrides: object) -> OpenAIEmbeddingProvider:
        return build(requests, **overrides)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def provider(requests: list[httpx.Request]) -> OpenAIEmbeddingProvider:
    return build(requests)


class TestOpenAIContract(EmbeddingProviderContract):
    """Every case in the shared suite, against the OpenAI driver."""


async def test_token_counts_come_from_the_api(provider: OpenAIEmbeddingProvider) -> None:
    # cl100k_base is the encoder the API bills with, so the reported usage and
    # our pre-flight count agree exactly rather than approximately.
    text = "the quick brown fox jumps over the lazy dog"

    batch = await provider.embed_documents([text])

    assert batch.token_source is TokenSource.PROVIDER
    assert batch.tokens == provider.count_tokens(text)


async def test_the_requested_dimensions_are_sent(
    provider: OpenAIEmbeddingProvider, requests: list[httpx.Request]
) -> None:
    # Without this the API returns its default width and the collection name
    # stops describing what is in the collection (AD-006).
    await provider.embed_documents(["alpha"])

    assert json.loads(requests[0].content)["dimensions"] == DIMENSIONS


async def test_documents_and_queries_reach_the_same_endpoint(
    provider: OpenAIEmbeddingProvider, requests: list[httpx.Request]
) -> None:
    # AD-007's asymmetry is a no-op for OpenAI. Asserted rather than assumed,
    # because "no task type" is a fact about the API, not an oversight.
    await provider.embed_documents(["alpha"])
    await provider.embed_query("alpha")

    document_body = json.loads(requests[0].content)
    query_body = json.loads(requests[1].content)
    assert requests[0].url == requests[1].url
    assert document_body.keys() == query_body.keys()


async def test_a_rate_limit_is_an_upstream_error(
    make_provider: Callable[..., OpenAIEmbeddingProvider],
) -> None:
    # 429 is the provider having a bad moment, not our misconfiguration.
    with pytest.raises(UpstreamError):
        await make_provider(status=429).embed_documents(["alpha"])


async def test_an_unreachable_provider_is_an_upstream_error(
    make_provider: Callable[..., OpenAIEmbeddingProvider],
) -> None:
    # A connection failure never becomes an `APIStatusError`, so it needs its
    # own arm; without one it escapes as a raw httpx exception and the single
    # problem+json handler has nothing to map.
    with pytest.raises(UpstreamError):
        await make_provider(unreachable=True).embed_documents(["alpha"])


async def test_a_wrong_width_response_is_an_upstream_error(
    make_provider: Callable[..., OpenAIEmbeddingProvider],
) -> None:
    # Caught here rather than at the Chroma upsert, which is after the tokens
    # are spent and after the Postgres rows are written.
    with pytest.raises(UpstreamError):
        await make_provider(width=512).embed_documents(["alpha"])


async def test_a_model_override_changes_the_collection_it_addresses() -> None:
    provider = OpenAIEmbeddingProvider(
        EmbeddingProviderSettings(api_key="sk-test", dimensions=512),
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    assert provider.model.dimensions == 512
    assert provider.model.collection_name(1).endswith("__512__c1")
