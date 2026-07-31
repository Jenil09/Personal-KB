"""Collection identity, which is what binds an index to one model (AD-006)."""

import pytest

from ai_embeddings import EmbeddingModel, EmbeddingProviderSettings

MODEL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=1536,
    max_input_tokens=8191,
    max_batch_inputs=96,
)


def test_the_collection_name_matches_the_design() -> None:
    assert MODEL.collection_name(1) == "kb__openai__text_embedding_3_small__1536__c1"


def test_the_slug_survives_a_model_id_with_punctuation() -> None:
    # Collection names go into Chroma, which is stricter about characters than
    # model IDs are.
    model = EmbeddingModel("gemini", "gemini-embedding-001", 1536, 2048, 96)

    assert model.slug == "gemini_embedding_001"
    assert model.collection_name(1) == "kb__gemini__gemini_embedding_001__1536__c1"


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"dimensions": 512}, "kb__openai__text_embedding_3_small__512__c1"),
        ({"model_id": "text-embedding-3-large"}, "kb__openai__text_embedding_3_large__1536__c1"),
    ],
)
def test_an_override_addresses_a_different_collection(
    override: dict[str, object], expected: str
) -> None:
    # Switching model or width is a re-embed migration, not a config flip. The
    # name changing is what stops one being mistaken for the other.
    changed = MODEL.overridden(
        model_id=override.get("model_id"),  # type: ignore[arg-type]
        dimensions=override.get("dimensions"),  # type: ignore[arg-type]
    )

    assert changed.collection_name(1) == expected


def test_no_override_returns_the_same_model() -> None:
    assert MODEL.overridden(None, None) is MODEL


def test_models_are_frozen() -> None:
    # A model mutated after a collection was created would silently start
    # naming a different one.
    with pytest.raises(AttributeError):
        MODEL.dimensions = 3072  # type: ignore[misc]


def test_the_api_key_has_no_default() -> None:
    # A missing key fails at startup, not on the first search.
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError, match="api_key"):
        EmbeddingProviderSettings()  # type: ignore[call-arg]


def test_the_api_key_stays_out_of_reprs() -> None:
    settings = EmbeddingProviderSettings(api_key="sk-do-not-log-me")

    assert "sk-do-not-log-me" not in repr(settings)
    assert settings.api_key.get_secret_value() == "sk-do-not-log-me"


def test_the_default_timeouts_are_the_designs() -> None:
    settings = EmbeddingProviderSettings(api_key="sk-test")

    assert settings.connect_timeout_seconds == 5.0
    assert settings.read_timeout_seconds == 30.0


def test_the_shared_client_carries_those_timeouts() -> None:
    # An embedding call is the slowest stage of a search and sits inside a
    # 1.5 s p95 target; an unbounded read timeout makes that target fiction.
    from ai_embeddings import create_http_client

    client = create_http_client(EmbeddingProviderSettings(api_key="sk-test"))

    assert client.timeout.connect == 5.0
    assert client.timeout.read == 30.0
