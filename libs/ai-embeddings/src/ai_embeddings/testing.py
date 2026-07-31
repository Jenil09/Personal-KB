"""The contract suite both drivers must pass (AD-006).

Ships inside the package rather than beside one driver's tests so there is
exactly one copy: a suite that lives in `tests/` gets forked the moment the
second driver is inconvenient, and a forked contract is not a contract. It also
means a service adding its own provider inherits the same bar.

A driver's test module subclasses `EmbeddingProviderContract` and supplies the
fixtures below. The stub responses must follow one convention: **the vector for
input `i` is the `i`-th standard basis vector** — all zeros but for a 1.0 at
position `i`. Two reasons. It is already unit length, so a driver that
re-normalises truncated output (Gemini does) passes it through unchanged; and
the position of the maximum identifies which input a vector came from, which is
how the ordering test catches a driver that returns vectors misaligned with the
texts that produced them.

Install with the `ai-embeddings[testing]` extra.
"""

from collections.abc import Callable, Sequence

import pytest

from ai_embeddings.port import EmbeddingProvider, TokenSource
from platform_core import ConfigurationError, UpstreamError, ValidationError

__all__ = ["EmbeddingProviderContract", "basis_vector"]


def basis_vector(index: int, dimensions: int) -> list[float]:
    """The `index`-th standard basis vector. Unit length, and self-identifying."""
    vector = [0.0] * dimensions
    vector[index % dimensions] = 1.0
    return vector


def _source_of(vector: Sequence[float]) -> int:
    """Which input produced this vector, by the basis-vector convention."""
    return max(range(len(vector)), key=vector.__getitem__)


class EmbeddingProviderContract:
    """Subclass as `class TestFooContract(EmbeddingProviderContract)`.

    Required fixtures:

    - `provider` — a driver wired to a stub that answers by the convention above
    - `requests` — the list the stub appends each outgoing request to, so a test
      can assert a request was *not* made
    - `make_provider` — `(status: int = 200, vectors: int | None = None)`,
      returning a driver whose stub answers with that HTTP status, or with that
      many vectors regardless of how many inputs were sent

    They are deliberately not declared here as overridable stubs: a fixture
    defined on the class shadows the module-level one the driver supplies, so
    the placeholder would win and every case would fail on it.
    """

    # --- shape -----------------------------------------------------------

    async def test_documents_get_one_vector_each(self, provider: EmbeddingProvider) -> None:
        batch = await provider.embed_documents(["alpha", "beta", "gamma"])

        assert len(batch) == 3

    async def test_vectors_have_the_models_dimensions(self, provider: EmbeddingProvider) -> None:
        # A width mismatch is only noticed when Chroma rejects the upsert,
        # which is after the tokens are spent.
        batch = await provider.embed_documents(["alpha"])

        assert len(batch.vectors[0]) == provider.model.dimensions

    async def test_vectors_stay_aligned_with_their_inputs(
        self, provider: EmbeddingProvider
    ) -> None:
        # Misalignment is silent: every chunk gets a vector, just the wrong
        # one, and the index is quietly wrong forever after.
        batch = await provider.embed_documents(["alpha", "beta", "gamma", "delta"])

        assert [_source_of(vector) for vector in batch.vectors] == [0, 1, 2, 3]

    async def test_a_query_returns_a_single_vector(self, provider: EmbeddingProvider) -> None:
        embedding = await provider.embed_query("what did I write about vector databases")

        assert len(embedding.vector) == provider.model.dimensions

    async def test_the_batch_carries_the_model_it_was_embedded_with(
        self, provider: EmbeddingProvider
    ) -> None:
        # A vector is meaningless without knowing which space it is in (AD-006).
        batch = await provider.embed_documents(["alpha"])

        assert batch.model == provider.model

    # --- tokens ----------------------------------------------------------

    async def test_token_counts_are_reported_and_attributed(
        self, provider: EmbeddingProvider
    ) -> None:
        batch = await provider.embed_documents(["alpha beta gamma"])

        assert batch.tokens > 0
        # Exact for OpenAI, estimated for Gemini — the caller writing this into
        # the audit trail needs to know which it has.
        assert batch.token_source in set(TokenSource)

    async def test_counting_tokens_needs_no_request(
        self, provider: EmbeddingProvider, requests: list[object]
    ) -> None:
        # Pre-flight budgeting happens before batching decisions, so it cannot
        # cost a round trip per candidate chunk.
        assert provider.count_tokens("alpha beta gamma") > 0
        assert requests == []

    # --- validation, before a request is spent ---------------------------

    async def test_nothing_to_embed_costs_nothing(
        self, provider: EmbeddingProvider, requests: list[object]
    ) -> None:
        # A re-ingest where every chunk hash already exists (AD-008) has an
        # empty list here. Zero-cost success, not an error.
        batch = await provider.embed_documents([])

        assert len(batch) == 0
        assert batch.tokens == 0
        assert requests == []

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    async def test_a_blank_input_is_rejected_locally(
        self, provider: EmbeddingProvider, requests: list[object], blank: str
    ) -> None:
        with pytest.raises(ValidationError):
            await provider.embed_documents(["alpha", blank])

        assert requests == []

    async def test_a_blank_query_is_rejected_locally(
        self, provider: EmbeddingProvider, requests: list[object]
    ) -> None:
        with pytest.raises(ValidationError):
            await provider.embed_query("  ")

        assert requests == []

    async def test_too_many_inputs_are_rejected_locally(
        self, provider: EmbeddingProvider, requests: list[object]
    ) -> None:
        oversized = ["alpha"] * (provider.model.max_batch_inputs + 1)

        with pytest.raises(ValidationError):
            await provider.embed_documents(oversized)

        assert requests == []

    async def test_an_input_over_the_context_window_is_rejected_locally(
        self, provider: EmbeddingProvider, requests: list[object]
    ) -> None:
        # Naming the offending input matters: the alternative is a 400 from the
        # provider that says only that the batch was bad.
        huge = "token " * (provider.model.max_input_tokens + 10)

        with pytest.raises(ValidationError):
            await provider.embed_documents([huge])

        assert requests == []

    # --- failure mapping -------------------------------------------------

    async def test_a_provider_outage_is_an_upstream_error(
        self, make_provider: Callable[..., EmbeddingProvider]
    ) -> None:
        failing = make_provider(status=503)

        with pytest.raises(UpstreamError):
            await failing.embed_documents(["alpha"])

    async def test_a_rejected_key_is_a_configuration_error(
        self, make_provider: Callable[..., EmbeddingProvider]
    ) -> None:
        # 401 from the provider is our misconfiguration, not their outage, and
        # retrying it forever would be the wrong response.
        failing = make_provider(status=401)

        with pytest.raises(ConfigurationError):
            await failing.embed_documents(["alpha"])

    async def test_a_short_response_is_an_upstream_error(
        self, make_provider: Callable[..., EmbeddingProvider]
    ) -> None:
        # Dropping an input silently is worse than failing: it shifts every
        # subsequent vector onto the wrong chunk.
        short = make_provider(vectors=2)

        with pytest.raises(UpstreamError):
            await short.embed_documents(["alpha", "beta", "gamma"])

    # --- identity --------------------------------------------------------

    def test_the_collection_name_is_derived_and_stable(self, provider: EmbeddingProvider) -> None:
        model = provider.model
        name = model.collection_name(1)

        assert name == f"kb__{model.provider}__{model.slug}__{model.dimensions}__c1"
        assert name == model.collection_name(1)
        # The chunker version is in the name because a chunker change forces a
        # full re-embed into a new collection (AD-008).
        assert model.collection_name(2) != name
