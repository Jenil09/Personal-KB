"""The Gemini driver against the shared contract, plus what is specific to it.

The task-type assertions are the reason AD-007 exists. Sending
`RETRIEVAL_DOCUMENT` for a query does not error — it silently returns worse
results — so the only place it can be caught is here, on the wire.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from ai_embeddings import EmbeddingProviderSettings, TokenSource
from ai_embeddings.adapters.gemini import GEMINI_EMBEDDING_001, GeminiEmbeddingProvider
from ai_embeddings.testing import EmbeddingProviderContract, basis_vector

DIMENSIONS = GEMINI_EMBEDDING_001.dimensions


def build(
    recorded: list[httpx.Request],
    *,
    status: int = 200,
    vectors: int | None = None,
    unnormalised: bool = False,
    zeroed: bool = False,
    native_width: bool = False,
) -> GeminiEmbeddingProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status, "message": "nope"}})

        body = json.loads(request.content)
        count = len(body.get("requests", [body])) if vectors is None else vectors
        scale = 4.0 if unnormalised else 1.0
        width = 3072 if native_width else DIMENSIONS
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {
                        "values": [0.0] * width
                        if zeroed
                        else [value * scale for value in basis_vector(index, width)]
                    }
                    for index in range(count)
                ]
            },
        )

    settings = EmbeddingProviderSettings(
        api_key="gemini-test",
        max_retries=0,
        dimensions=3072 if native_width else None,
    )
    return GeminiEmbeddingProvider(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def make_provider(requests: list[httpx.Request]) -> Callable[..., GeminiEmbeddingProvider]:
    def factory(**overrides: object) -> GeminiEmbeddingProvider:
        return build(requests, **overrides)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def provider(requests: list[httpx.Request]) -> GeminiEmbeddingProvider:
    return build(requests)


class TestGeminiContract(EmbeddingProviderContract):
    """Every case in the shared suite, against the Gemini driver.

    Identical to the OpenAI class body on purpose: that the two drivers differ
    nowhere the port can see is the claim AD-006 rests on.
    """


def task_types(request: httpx.Request) -> list[str]:
    return [item["taskType"] for item in json.loads(request.content)["requests"]]


async def test_ingest_sends_the_document_task_type(
    provider: GeminiEmbeddingProvider, requests: list[httpx.Request]
) -> None:
    await provider.embed_documents(["alpha", "beta"])

    assert task_types(requests[0]) == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_DOCUMENT"]


async def test_search_sends_the_query_task_type(
    provider: GeminiEmbeddingProvider, requests: list[httpx.Request]
) -> None:
    await provider.embed_query("what did I write about vector databases")

    assert task_types(requests[0]) == ["RETRIEVAL_QUERY"]


async def test_the_two_methods_are_distinguishable_on_the_wire(
    provider: GeminiEmbeddingProvider, requests: list[httpx.Request]
) -> None:
    # The whole point of AD-007: a single `embed()` would make these identical
    # and nothing would ever notice.
    await provider.embed_documents(["alpha"])
    await provider.embed_query("alpha")

    assert task_types(requests[0]) != task_types(requests[1])


async def test_the_truncated_width_is_requested(
    provider: GeminiEmbeddingProvider, requests: list[httpx.Request]
) -> None:
    await provider.embed_documents(["alpha"])

    sent = json.loads(requests[0].content)["requests"][0]
    assert sent["outputDimensionality"] == DIMENSIONS


async def test_truncated_vectors_are_renormalised(
    make_provider: Callable[..., GeminiEmbeddingProvider],
) -> None:
    # Only the native 3072-wide output is unit length. The collection is
    # cosine-space (Design §2.3), so an unnormalised vector distorts every
    # distance it takes part in.
    provider = make_provider(unnormalised=True)

    batch = await provider.embed_documents(["alpha"])

    magnitude = sum(value * value for value in batch.vectors[0]) ** 0.5
    assert magnitude == pytest.approx(1.0)


async def test_native_width_output_is_left_alone(
    make_provider: Callable[..., GeminiEmbeddingProvider],
) -> None:
    # The model already normalises its full 3072-wide output. Normalising it
    # again would be a second rounding pass over every vector for nothing.
    provider = make_provider(native_width=True, unnormalised=True)

    batch = await provider.embed_documents(["alpha"])

    assert max(batch.vectors[0]) == 4.0


async def test_a_zero_vector_is_returned_rather_than_divided_by(
    make_provider: Callable[..., GeminiEmbeddingProvider],
) -> None:
    # Degenerate, but a ZeroDivisionError here would surface as a 500 on a
    # search rather than as anything diagnosable.
    provider = make_provider(zeroed=True)

    batch = await provider.embed_documents(["alpha"])

    assert set(batch.vectors[0]) == {0.0}


async def test_token_counts_are_marked_as_estimates(
    provider: GeminiEmbeddingProvider,
) -> None:
    # Gemini's embedding response carries a billable character count and no
    # token count, so this figure is tiktoken against the wrong tokeniser. It
    # is fine for batching and is not a billing number.
    batch = await provider.embed_documents(["alpha beta gamma"])

    assert batch.token_source is TokenSource.ESTIMATED
    assert batch.tokens > 0


async def test_the_two_drivers_address_different_collections() -> None:
    # A Gemini vector never belongs in the OpenAI collection, whatever their
    # widths (AD-006). The name is what enforces that.
    from ai_embeddings.adapters.openai import TEXT_EMBEDDING_3_SMALL

    assert GEMINI_EMBEDDING_001.collection_name(1) != TEXT_EMBEDDING_3_SMALL.collection_name(1)
    assert GEMINI_EMBEDDING_001.collection_name(1).startswith("kb__gemini__")
