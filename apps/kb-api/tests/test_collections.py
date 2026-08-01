"""Collection naming — the guard that keeps incomparable vectors apart."""

import pytest

from kb_api.chunking import CHUNKER_VERSION
from kb_api.domain import collection_name, model_slug


def test_the_v1_name_matches_the_design():
    assert (
        collection_name(
            provider="openai",
            model_id="text-embedding-3-small",
            dimensions=1536,
            chunker_version=1,
        )
        == "kb__openai__text_embedding_3_small__1536__c1"
    )


def test_a_namespaced_model_id_slugs_to_one_underscore_per_run():
    assert model_slug("models/gemini-embedding-001") == "models_gemini_embedding_001"


def test_every_component_changes_the_name():
    variants = {
        collection_name(
            provider=provider, model_id=model_id, dimensions=dimensions, chunker_version=version
        )
        for provider, model_id, dimensions, version in [
            ("openai", "text-embedding-3-small", 1536, 1),
            ("gemini", "text-embedding-3-small", 1536, 1),
            ("openai", "text-embedding-3-large", 1536, 1),
            ("openai", "text-embedding-3-small", 3072, 1),
            ("openai", "text-embedding-3-small", 1536, 2),
        ]
    }
    assert len(variants) == 5


def test_the_current_chunker_version_is_usable_as_a_component():
    name = collection_name(
        provider="openai",
        model_id="text-embedding-3-small",
        dimensions=1536,
        chunker_version=CHUNKER_VERSION,
    )
    assert name.endswith(f"__c{CHUNKER_VERSION}")


@pytest.mark.parametrize(
    ("model_id", "dimensions", "chunker_version"),
    [
        ("text-embedding-3-small", 0, 1),
        ("text-embedding-3-small", 1536, 0),
        ("///", 1536, 1),
    ],
)
def test_a_name_that_could_not_identify_a_space_is_rejected(model_id, dimensions, chunker_version):
    with pytest.raises(ValueError):
        collection_name(
            provider="openai",
            model_id=model_id,
            dimensions=dimensions,
            chunker_version=chunker_version,
        )
