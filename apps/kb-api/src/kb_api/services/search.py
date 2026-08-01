"""Search — Technical Design §3.2.

The flow: resolve the collection, refuse an unpopulated one, resolve tag filters
through Postgres (AD-005), build the `where` clause, embed the query (cached,
AD-008), one Chroma query, map distances to scores. Postgres is otherwise off
the path entirely — the response is built from the Chroma payload alone (AD-004).

Three things here are not in §3.2's numbered list.

**The `409` is decided before the query is embedded.** §3.2 resolves the
collection at step 1 and embeds at step 2, which already implies this ordering,
but it is worth stating why it must not be relaxed: a collection nobody has
ingested into is the *expected* state of a freshly configured second provider,
and answering it with an embedding call means paying a provider round trip to
find out there was nothing to search.

**The tag pre-filter runs before the embedding too** — §3.2 has it at step 3.
When no live document carries the requested tags the answer is an empty result
set no matter what the query vector is, so embedding first would spend a call on
a search that cannot match. It is observable only as `query_tokens: 0` on that
response, which is the honest number: nothing was embedded.

**An empty pre-filter result short-circuits rather than becoming `$in: []`.**
An empty `$in` is not "match nothing" to reason about — it is a clause whose
behaviour depends on the store, and Design §3.2 gives it no meaning. The service
answers with no results and never issues the query.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from kb_api.domain import (
    DOCUMENT_ID_KEY,
    DocumentFilter,
    DocumentStore,
    MatchMetadata,
    VectorMatch,
    VectorStore,
    read_metadata,
)
from kb_api.services.providers import ProviderRegistry, ResolvedProvider
from kb_api.services.query_cache import CachedQuery, QueryEmbeddingCache, query_cache_key
from platform_core import ConflictError, ValidationError, get_logger
from platform_db import SessionSource

__all__ = [
    "SearchFilters",
    "SearchHit",
    "SearchQuery",
    "SearchResult",
    "SearchService",
    "build_where",
    "score_of",
]

_logger = get_logger("kb.search")


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """PRD §6.2's `filters`. Scalars go to Chroma, tags go to Postgres (AD-005)."""

    type: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    match_all_tags: bool = False

    @property
    def has_tags(self) -> bool:
        return bool(self.tags)


@dataclass(frozen=True, slots=True)
class SearchQuery:
    query: str
    top_k: int = 5
    provider: str | None = None
    filters: SearchFilters = SearchFilters()


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result. `score` is a similarity; `VectorMatch` carried a distance."""

    id: str
    text: str
    metadata: MatchMetadata
    score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What §6.2 answers with, plus what tier-2 telemetry will record (Phase 8).

    `cached` and `embedding_calls` are not in the PRD response body. They are
    here for the same reason `IngestResult.embedding_calls` is: a test asserting
    the cache works should read a counter rather than infer it from a timing.
    """

    hits: tuple[SearchHit, ...]
    collection: str
    latency_ms: int
    query_tokens: int = 0
    cached: bool = False
    embedding_calls: int = 0


def score_of(distance: float) -> float:
    """PRD §6.2: cosine similarity as `1 - distance`.

    Not clamped. Chroma's cosine distance runs 0-2, so a chunk pointing away
    from the query legitimately scores negative, and flattening that to zero
    would make "unrelated" and "opposite" indistinguishable. Collections are
    created cosine-spaced (Design §2.3) and the adapter warns when it finds one
    that is not, which is the case where this stops being a similarity at all.
    """
    return 1.0 - distance


def build_where(
    filters: SearchFilters, document_ids: Sequence[str] | None
) -> Mapping[str, object] | None:
    """§3.2 step 4. `None` when nothing is filtered — never `{}`.

    Chroma reads an empty clause as matching nothing, so the difference between
    "no filter" and "a filter with no terms" is the difference between every
    result and none.

    Multiple terms are wrapped in `$and` because Chroma rejects a clause with
    two keys at the top level.
    """
    clauses: list[Mapping[str, object]] = []
    if filters.type is not None:
        clauses.append({"type": {"$eq": filters.type}})
    if filters.source is not None:
        clauses.append({"source": {"$eq": filters.source}})
    if document_ids is not None:
        clauses.append({DOCUMENT_ID_KEY: {"$in": list(document_ids)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class SearchService:
    def __init__(
        self,
        *,
        sessions: SessionSource,
        documents: DocumentStore,
        vectors: VectorStore,
        providers: ProviderRegistry,
        cache: QueryEmbeddingCache,
        tag_filter_limit: int,
    ) -> None:
        self._sessions = sessions
        self._documents = documents
        self._vectors = vectors
        self._providers = providers
        self._cache = cache
        self._tag_filter_limit = tag_filter_limit

    async def search(self, request: SearchQuery) -> SearchResult:
        started = perf_counter()
        target = self._providers.resolve(request.provider)
        query = request.query.strip()
        if not query:
            raise ValidationError("Search query is empty.")

        await self._require_populated(target)

        document_ids = await self._tag_prefilter(target, request.filters)
        if document_ids is not None and not document_ids:
            _logger.info(
                "search_no_tag_matches",
                collection=target.collection,
                tags=list(request.filters.tags),
            )
            return SearchResult(
                hits=(), collection=target.collection, latency_ms=_elapsed_ms(started)
            )

        embedded, cached = await self._embed_query(target, query)
        matches = await self._vectors.query(
            target.collection,
            embedded.vector,
            top_k=request.top_k,
            where=build_where(request.filters, document_ids),
        )

        result = SearchResult(
            hits=tuple(_to_hit(match) for match in matches),
            collection=target.collection,
            latency_ms=_elapsed_ms(started),
            query_tokens=embedded.tokens,
            cached=cached,
            embedding_calls=0 if cached else 1,
        )
        _logger.info(
            "search_complete",
            collection=target.collection,
            top_k=request.top_k,
            results=len(result.hits),
            filtered=document_ids is not None,
            cached=cached,
            latency_ms=result.latency_ms,
        )
        return result

    async def _require_populated(self, target: ResolvedProvider) -> None:
        """§3.2 step 1. An empty or absent collection is a `409`, not no results.

        One `count` answers both: the adapter reports zero for a collection that
        was never created, and a collection that exists with nothing in it is
        the same thing from a caller's point of view — the provider was named
        correctly and there is nothing indexed under it. An unconfigured
        provider is the registry's `422`, a different failure with a different
        fix (`providers.py`).
        """
        if await self._vectors.count(target.collection) == 0:
            raise ConflictError(
                f"No documents are indexed for provider {target.name!r}.",
                context={"provider": target.name, "collection": target.collection},
            )

    async def _tag_prefilter(
        self, target: ResolvedProvider, filters: SearchFilters
    ) -> tuple[str, ...] | None:
        """AD-005: tags resolve to a document ID set through Postgres.

        `None` means no tag filter was asked for, which is not the same as the
        empty tuple — that one means tags were asked for and nothing matched.

        Tags only. `type` and `source` are answerable from Chroma's metadata and
        go straight into the `where` clause; adding them here as well would make
        the result depend on the two stores agreeing about a document's type,
        for no gain at a corpus this size.

        Scoped to the target collection: a document ingested under a different
        provider has no vectors here, and letting its ID into the `$in` clause
        would grow the clause with terms that cannot match.
        """
        if not filters.has_tags:
            return None
        async with self._sessions.session() as session:
            ids = await self._documents.ids_matching(
                session,
                DocumentFilter(
                    collection=target.collection,
                    tags=filters.tags,
                    match_all_tags=filters.match_all_tags,
                ),
                limit=self._tag_filter_limit,
            )
        if len(ids) == self._tag_filter_limit:
            # AD-005 flagged the `$in` clause getting unwieldy past a few
            # thousand IDs. The cap is never reached at personal-KB scale, so
            # reaching it is worth a line in the log rather than a silent
            # truncation nobody can distinguish from a genuine result.
            _logger.warning(
                "tag_prefilter_truncated",
                collection=target.collection,
                limit=self._tag_filter_limit,
            )
        return tuple(str(identifier) for identifier in ids)

    async def _embed_query(self, target: ResolvedProvider, query: str) -> tuple[CachedQuery, bool]:
        """§3.2 step 2 — the cache, then `embed_query` (AD-007) on a miss."""
        key = query_cache_key(query, target.collection)
        hit = self._cache.get(key)
        if hit is not None:
            return hit, True

        embedding = await target.provider.embed_query(query)
        entry = CachedQuery(
            vector=embedding.vector,
            tokens=embedding.tokens,
            token_source=embedding.token_source,
        )
        self._cache.put(key, entry)
        return entry, False


def _to_hit(match: VectorMatch) -> SearchHit:
    return SearchHit(
        id=match.id,
        text=match.chunk_text,
        metadata=read_metadata(match.metadata),
        score=score_of(match.distance),
    )


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)
