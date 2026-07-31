"""The one claim a stub cannot make: our token counts match OpenAI's billing.

Every other test here asserts the driver's behaviour against a transport we
control, which means the usage number in the response is one we wrote. That
proves the driver reads it, not that `tiktoken` agrees with the API — and
"agrees with the API" is the Phase 3 exit criterion.

Skipped unless `OPENAI_API_KEY` is set. It costs a few hundred tokens (well
under a cent) and is the reason the pre-flight budget can be trusted.
"""

import os

import pytest

from ai_embeddings import EmbeddingProviderSettings, TokenSource, create_http_client
from ai_embeddings.adapters.openai import OpenAIEmbeddingProvider

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="needs a real OPENAI_API_KEY to compare against the API's reported usage",
    ),
]

# Deliberately awkward: contractions, code, punctuation, unicode, and a long
# technical run — the places a naive tokeniser and the real one diverge.
FIXTURES = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "def embed_documents(texts: Sequence[str]) -> EmbeddingBatch: ...",
    "PostgreSQL's `jsonb_path_ops` GIN index — 5 ms, indexed, AD-005.",
    "Ich schreibe über Vektordatenbanken und Einbettungen.",
    "supercalifragilisticexpialidocious " * 20,
]


@pytest.fixture
async def provider():
    settings = EmbeddingProviderSettings(api_key=os.environ["OPENAI_API_KEY"])
    client = create_http_client(settings)
    try:
        yield OpenAIEmbeddingProvider(settings, client)
    finally:
        await client.aclose()


@pytest.mark.parametrize("text", FIXTURES)
async def test_our_count_matches_the_apis_reported_usage(provider, text: str) -> None:
    batch = await provider.embed_documents([text])

    assert batch.token_source is TokenSource.PROVIDER
    assert batch.tokens == provider.count_tokens(text)


async def test_a_multi_input_batch_totals_correctly(provider) -> None:
    # Per-input agreement does not imply the batch total agrees; the API bills
    # the request, and this is the number that lands in the audit trail.
    batch = await provider.embed_documents(FIXTURES)

    assert batch.tokens == sum(provider.count_tokens(text) for text in FIXTURES)


async def test_the_real_api_returns_the_width_we_asked_for(provider) -> None:
    batch = await provider.embed_documents(["alpha"])

    assert len(batch.vectors[0]) == provider.model.dimensions
