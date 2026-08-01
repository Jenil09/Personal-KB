"""The search flow's decisions, against fakes (Design §3.2).

Unit, not integration. What is asserted here is what the service *decides* —
which clause it builds, when it declines to spend an embedding call, what it
does with a tag filter that matches nothing — and every one of those is visible
at the port. The behaviour that needs a real Chroma is the behaviour that
depends on Chroma's own semantics: ranking, and whether a `where` clause filters
the way the naive equivalent does. That lives in `test_search_api.py`.

The fakes are cast to their ports rather than implementing them in full. A fake
that satisfied every method of `DocumentStore` would be mostly `NotImplemented`
bodies asserting nothing.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import cast
from uuid import UUID, uuid4

import pytest

from ai_embeddings import EmbeddingModel, EmbeddingProvider, RawEmbeddings
from ai_embeddings.port import Purpose, TokenSource
from kb_api.domain import (
    DOCUMENT_ID_KEY,
    ChunkMetadata,
    DocumentFilter,
    DocumentStore,
    VectorMatch,
    VectorStore,
)
from kb_api.services import ProviderRegistry, QueryEmbeddingCache, SearchService
from kb_api.services.search import SearchFilters, SearchQuery, build_where, score_of
from platform_core import ConflictError, ValidationError
from platform_db import SessionSource

DIMENSIONS = 4

MODEL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=DIMENSIONS,
    max_input_tokens=8191,
    max_batch_inputs=96,
)


class CountingProvider(EmbeddingProvider):
    """Records every query it was asked to embed, so the cache is checkable."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    @property
    def model(self) -> EmbeddingModel:
        return MODEL

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        # AD-007: a search must reach the provider on the query side. A driver
        # sent `RETRIEVAL_DOCUMENT` for a query degrades Gemini silently.
        assert purpose is Purpose.QUERY
        self.queries.extend(texts)
        return RawEmbeddings(
            tuple((1.0, 0.0, 0.0, 0.0) for _ in texts),
            sum(self.count_tokens(text) for text in texts),
            TokenSource.PROVIDER,
        )


class FakeVectorStore:
    """Records what it was queried with; answers with whatever it was given."""

    def __init__(self, *, count: int = 3, matches: tuple[VectorMatch, ...] = ()) -> None:
        self._count = count
        self._matches = matches
        self.queries: list[Mapping[str, object] | None] = []
        self.top_ks: list[int] = []

    async def count(self, collection: str) -> int:
        return self._count

    async def query(
        self,
        collection: str,
        vector: tuple[float, ...],
        *,
        top_k: int,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorMatch, ...]:
        self.queries.append(where)
        self.top_ks.append(top_k)
        return self._matches


class FakeDocumentStore:
    """The AD-005 tag lookup, and nothing else the search path calls."""

    def __init__(self, ids: tuple[UUID, ...] = ()) -> None:
        self._ids = ids
        self.filters: list[DocumentFilter] = []

    async def ids_matching(
        self, session: object, filters: DocumentFilter, *, limit: int | None = None
    ) -> tuple[UUID, ...]:
        self.filters.append(filters)
        return self._ids


class FakeSessions:
    def __init__(self) -> None:
        self.opened = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[object]:
        self.opened += 1
        yield object()


def match(identifier: str, distance: float, **metadata: str | int | float | bool) -> VectorMatch:
    payload: ChunkMetadata = {
        DOCUMENT_ID_KEY: str(uuid4()),
        "title": "Redshift Architecture",
        "type": "architecture",
        "tags": "ansible|hardening",
        "ordinal": 3,
    }
    payload.update(metadata)
    return VectorMatch(id=identifier, chunk_text="body", metadata=payload, distance=distance)


def build(
    *,
    vectors: FakeVectorStore | None = None,
    documents: FakeDocumentStore | None = None,
    provider: CountingProvider | None = None,
    sessions: FakeSessions | None = None,
    cache_size: int = 8,
) -> SearchService:
    driver = provider or CountingProvider()
    return SearchService(
        sessions=cast("SessionSource", sessions or FakeSessions()),
        documents=cast("DocumentStore", documents or FakeDocumentStore()),
        vectors=cast("VectorStore", vectors or FakeVectorStore()),
        providers=ProviderRegistry({"openai": driver}, default="openai", chunker_version=1),
        cache=QueryEmbeddingCache(max_entries=cache_size, ttl_seconds=900.0),
        tag_filter_limit=2000,
    )


# --- score mapping -------------------------------------------------------


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (2.0, -1.0)],
)
def test_a_distance_maps_to_one_minus_itself(distance: float, expected: float) -> None:
    assert score_of(distance) == pytest.approx(expected)


def test_an_opposed_vector_scores_negative_rather_than_zero() -> None:
    # Chroma's cosine distance runs 0-2. Clamping would make "unrelated" and
    # "opposite" indistinguishable in the response.
    assert score_of(1.7) < 0


# --- where construction, §3.2 step 4 -------------------------------------


def test_no_filters_build_no_clause() -> None:
    # `None`, never `{}` — Chroma reads an empty clause as matching nothing.
    assert build_where(SearchFilters(), None) is None


def test_a_scalar_filter_goes_straight_into_the_clause() -> None:
    assert build_where(SearchFilters(type="architecture"), None) == {
        "type": {"$eq": "architecture"}
    }


def test_tag_results_arrive_as_an_id_set() -> None:
    assert build_where(SearchFilters(tags=("ansible",)), ["a", "b"]) == {
        DOCUMENT_ID_KEY: {"$in": ["a", "b"]}
    }


def test_two_terms_are_wrapped_in_and() -> None:
    # Chroma rejects a clause with two keys at the top level.
    clause = build_where(SearchFilters(type="architecture", tags=("ansible",)), ["a"])

    assert clause == {
        "$and": [{"type": {"$eq": "architecture"}}, {DOCUMENT_ID_KEY: {"$in": ["a"]}}]
    }


def test_source_is_filtered_in_chroma_not_in_postgres() -> None:
    # AD-005: scalars are metadata, so they need no Postgres round trip.
    assert build_where(SearchFilters(source="redshift.md"), None) == {
        "source": {"$eq": "redshift.md"}
    }


# --- the unpopulated collection, §3.2 step 1 -----------------------------


async def test_an_unpopulated_collection_is_a_409() -> None:
    service = build(vectors=FakeVectorStore(count=0))

    with pytest.raises(ConflictError):
        await service.search(SearchQuery(query="cilium"))


async def test_an_unpopulated_collection_costs_no_embedding_call() -> None:
    # The expected state of a freshly configured second provider. Paying a
    # provider round trip to discover there is nothing to search is the whole
    # reason the check comes first.
    provider = CountingProvider()
    service = build(vectors=FakeVectorStore(count=0), provider=provider)

    with pytest.raises(ConflictError):
        await service.search(SearchQuery(query="cilium"))

    assert provider.queries == []


async def test_an_unknown_provider_is_a_422_not_a_409() -> None:
    # Different failures with different fixes: a missing API key should not
    # look like an empty corpus (providers.py).
    service = build()

    with pytest.raises(ValidationError):
        await service.search(SearchQuery(query="cilium", provider="cohere"))


async def test_a_blank_query_is_rejected_before_anything_is_spent() -> None:
    provider = CountingProvider()
    service = build(provider=provider)

    with pytest.raises(ValidationError):
        await service.search(SearchQuery(query="   "))

    assert provider.queries == []


# --- the tag pre-filter, AD-005 ------------------------------------------


async def test_tags_are_resolved_against_the_targets_collection() -> None:
    documents = FakeDocumentStore(ids=(uuid4(),))
    service = build(documents=documents)

    await service.search(SearchQuery(query="cilium", filters=SearchFilters(tags=("ansible",))))

    applied = documents.filters[0]
    assert applied.tags == ("ansible",)
    assert applied.collection == "kb__openai__text_embedding_3_small__4__c1"


async def test_a_search_without_tags_never_touches_postgres() -> None:
    # AD-004: the common n8n case stays single-hop.
    sessions = FakeSessions()
    service = build(sessions=sessions)

    await service.search(SearchQuery(query="cilium"))

    assert sessions.opened == 0


async def test_tags_matching_no_document_return_nothing_without_querying_chroma() -> None:
    vectors = FakeVectorStore()
    provider = CountingProvider()
    service = build(vectors=vectors, documents=FakeDocumentStore(ids=()), provider=provider)

    result = await service.search(
        SearchQuery(query="cilium", filters=SearchFilters(tags=("nonexistent",)))
    )

    assert result.hits == ()
    assert vectors.queries == []
    # Nothing was sent to the provider, so nothing was spent — and the
    # response says so rather than reporting a count it did not pay for.
    assert provider.queries == []
    assert result.query_tokens == 0


async def test_match_all_tags_is_carried_through_to_the_lookup() -> None:
    documents = FakeDocumentStore(ids=(uuid4(),))
    service = build(documents=documents)

    await service.search(
        SearchQuery(query="cilium", filters=SearchFilters(tags=("a", "b"), match_all_tags=True))
    )

    assert documents.filters[0].match_all_tags is True


# --- the query cache, AD-008 ---------------------------------------------


async def test_a_repeated_query_embeds_once() -> None:
    provider = CountingProvider()
    service = build(provider=provider)

    first = await service.search(SearchQuery(query="cilium"))
    second = await service.search(SearchQuery(query="cilium"))

    assert provider.queries == ["cilium"]
    assert (first.cached, second.cached) == (False, True)
    assert (first.embedding_calls, second.embedding_calls) == (1, 0)


async def test_a_cache_hit_reports_the_token_count_it_was_stored_with() -> None:
    # Repeating a search should not change its answer. Actual spend is
    # `embedding_calls`, which is what tier-2 telemetry will record (Phase 8).
    service = build()

    first = await service.search(SearchQuery(query="cilium in kube-proxy replacement mode"))
    second = await service.search(SearchQuery(query="cilium in kube-proxy replacement mode"))

    assert first.query_tokens > 0
    assert second.query_tokens == first.query_tokens


async def test_a_different_query_is_a_different_entry() -> None:
    provider = CountingProvider()
    service = build(provider=provider)

    await service.search(SearchQuery(query="cilium"))
    await service.search(SearchQuery(query="longhorn"))

    assert provider.queries == ["cilium", "longhorn"]


async def test_the_cache_key_ignores_surrounding_whitespace() -> None:
    provider = CountingProvider()
    service = build(provider=provider)

    await service.search(SearchQuery(query="cilium"))
    await service.search(SearchQuery(query="  cilium  "))

    assert provider.queries == ["cilium"]


async def test_a_disabled_cache_embeds_every_time() -> None:
    provider = CountingProvider()
    service = build(provider=provider, cache_size=0)

    await service.search(SearchQuery(query="cilium"))
    await service.search(SearchQuery(query="cilium"))

    assert provider.queries == ["cilium", "cilium"]


# --- the response, §3.2 step 6 -------------------------------------------


async def test_hits_carry_the_metadata_chroma_returned() -> None:
    # AD-004: everything the response needs comes back in the query payload.
    vectors = FakeVectorStore(matches=(match("chunk-1", 0.2, title="Redshift"),))
    service = build(vectors=vectors)

    result = await service.search(SearchQuery(query="cilium"))

    hit = result.hits[0]
    assert hit.id == "chunk-1"
    assert hit.metadata.title == "Redshift"
    assert hit.metadata.tags == ("ansible", "hardening")
    assert hit.score == pytest.approx(0.8)


async def test_top_k_is_passed_through_to_the_store() -> None:
    vectors = FakeVectorStore()
    service = build(vectors=vectors)

    await service.search(SearchQuery(query="cilium", top_k=17))

    assert vectors.top_ks == [17]


async def test_a_result_reports_the_collection_it_came_from() -> None:
    service = build()

    result = await service.search(SearchQuery(query="cilium"))

    assert result.collection == "kb__openai__text_embedding_3_small__4__c1"
