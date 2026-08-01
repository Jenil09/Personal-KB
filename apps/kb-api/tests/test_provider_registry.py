"""Provider name to collection (AD-006).

A `provider` in a request body selects an index, not an algorithm. The two
things worth pinning are that the name it produces is the one Design §2.3
specifies, and that an unconfigured name fails as a `422` rather than as the
`409` search uses for an unpopulated collection — an operator whose API key is
missing should not see the same answer as an empty corpus.
"""

from collections.abc import Sequence

import pytest

from ai_embeddings import EmbeddingModel, EmbeddingProvider, RawEmbeddings
from ai_embeddings.port import Purpose
from kb_api.services import ProviderRegistry
from platform_core import ValidationError


class FakeProvider(EmbeddingProvider):
    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    def count_tokens(self, text: str) -> int:
        return len(text)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        raise AssertionError("the registry must never embed anything")


def provider(name: str, model_id: str, dimensions: int) -> FakeProvider:
    return FakeProvider(
        EmbeddingModel(
            provider=name,
            model_id=model_id,
            dimensions=dimensions,
            max_input_tokens=8191,
            max_batch_inputs=96,
        )
    )


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry(
        {
            "openai": provider("openai", "text-embedding-3-small", 1536),
            "gemini": provider("gemini", "models/gemini-embedding-001", 1536),
        },
        default="openai",
        chunker_version=1,
    )


def test_the_collection_name_is_the_one_the_design_specifies(registry: ProviderRegistry) -> None:
    assert registry.resolve("openai").collection == "kb__openai__text_embedding_3_small__1536__c1"


def test_a_model_id_with_a_slash_still_produces_a_legal_name(
    registry: ProviderRegistry,
) -> None:
    # `models/gemini-embedding-001` is what the provider calls it, and Chroma
    # will not take a slash in a collection name.
    assert (
        registry.resolve("gemini").collection == "kb__gemini__models_gemini_embedding_001__1536__c1"
    )


def test_no_provider_named_means_the_configured_default(registry: ProviderRegistry) -> None:
    assert registry.resolve(None) is registry.default
    assert registry.default.name == "openai"


def test_an_unconfigured_provider_is_a_422_not_a_409(registry: ProviderRegistry) -> None:
    with pytest.raises(ValidationError) as raised:
        registry.resolve("cohere")

    assert raised.value.status_code == 422
    assert raised.value.context["configured"] == ["openai", "gemini"]


def test_a_registry_with_no_providers_is_a_configuration_bug() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ProviderRegistry({}, default="openai", chunker_version=1)


def test_a_default_naming_an_absent_provider_fails_at_construction() -> None:
    # At startup, where it is cheap to fix — not on the first request that
    # omits `provider`.
    with pytest.raises(ValueError, match="default provider"):
        ProviderRegistry(
            {"gemini": provider("gemini", "gemini-embedding-001", 1536)},
            default="openai",
            chunker_version=1,
        )


def test_the_chunker_version_changes_the_collection() -> None:
    # AD-008: a chunker change invalidates every hash, so the vectors have to
    # land somewhere else rather than mix with the old ones.
    first = ProviderRegistry(
        {"openai": provider("openai", "text-embedding-3-small", 1536)},
        default="openai",
        chunker_version=1,
    )
    second = ProviderRegistry(
        {"openai": provider("openai", "text-embedding-3-small", 1536)},
        default="openai",
        chunker_version=2,
    )

    assert first.default.collection != second.default.collection
