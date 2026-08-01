"""The bounded query-embedding cache (AD-008, Design §3.2 step 2).

Technical Design §8 already reassessed this honestly: the n8n workload embeds a
different job description every run, so the steady-state hit rate is near zero.
The cache stays because it costs nothing idle and absorbs retries and manual
re-testing — not because it is a latency optimisation. That is why it is a few
dozen lines with no background sweeper and no metrics beyond two counters.

**The key includes the collection, not just the model id.** AD-008 wrote it as
`sha256(query + model_id)`, which is one identifier short: `dimensions` is an
operator override (`EmbeddingProviderSettings`), so the same `model_id` at 1536
and at 512 produces vectors of different widths in different spaces. Keying on
the collection name covers provider, model, dimensions, and chunker version at
once, because that is exactly what the name encodes (Design §2.3). A collision
here would return a vector from the wrong space and the search would answer with
plausible nonsense. **AD-021**

Expiry is lazy. Entries are checked on read and evicted by recency on write, so
a stale entry occupies a slot until it is either read or pushed out — bounded by
`max_entries` and therefore not a leak. A sweeper would be a background task
that exists to reclaim a few kilobytes.

No lock. The service runs one Uvicorn worker (AD-015) and every method here is
synchronous with no `await` inside, so nothing can interleave mid-update.
"""

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from ai_embeddings import EmbeddingVector
from ai_embeddings.port import TokenSource

__all__ = ["CachedQuery", "QueryEmbeddingCache", "query_cache_key"]


def query_cache_key(query: str, collection: str) -> str:
    """`sha256(query + collection)` — AD-021's amendment to AD-008's key.

    The separator is a NUL because neither a query nor a collection name can
    contain one, so no pair of inputs can concatenate to the same string.
    """
    return hashlib.sha256(f"{query}\0{collection}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedQuery:
    """A query's vector and what it cost to produce the first time."""

    vector: EmbeddingVector
    tokens: int
    token_source: TokenSource


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Exposed on the Phase 8 stats endpoint; counted from the start."""

    hits: int
    misses: int
    entries: int
    capacity: int


class QueryEmbeddingCache:
    """An LRU with a TTL, holding at most `max_entries` vectors.

    `max_entries=0` disables it: every lookup misses and nothing is stored, so
    an operator can turn the cache off without the call sites growing a branch.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries cannot be negative")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        # Monotonic deadlines, not wall-clock: a system clock stepped backwards
        # would otherwise keep an expired vector alive indefinitely.
        self._entries: OrderedDict[str, tuple[CachedQuery, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CachedQuery | None:
        found = self._entries.get(key)
        if found is None:
            self._misses += 1
            return None
        entry, expires_at = found
        if self._clock() >= expires_at:
            del self._entries[key]
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry

    def put(self, key: str, entry: CachedQuery) -> None:
        if self._max_entries == 0:
            return
        self._entries[key] = (entry, self._clock() + self._ttl)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            entries=len(self._entries),
            capacity=self._max_entries,
        )
