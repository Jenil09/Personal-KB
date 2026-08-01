"""The bounded query-embedding cache (AD-008, AD-021).

A unit suite with a fake clock. Testing a TTL by sleeping would either make the
suite slow or make the assertion depend on how loaded the machine is, and the
cache takes a `clock` for exactly that reason.
"""

from ai_embeddings.port import TokenSource
from kb_api.services.query_cache import CachedQuery, QueryEmbeddingCache, query_cache_key


class FakeClock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def entry(value: float = 1.0) -> CachedQuery:
    return CachedQuery(vector=(value, 0.0, 0.0), tokens=7, token_source=TokenSource.PROVIDER)


def cache(
    *, max_entries: int = 4, ttl_seconds: float = 60.0
) -> tuple[QueryEmbeddingCache, FakeClock]:
    clock = FakeClock()
    return QueryEmbeddingCache(max_entries=max_entries, ttl_seconds=ttl_seconds, clock=clock), clock


# --- the key -------------------------------------------------------------


def test_the_same_query_in_the_same_collection_is_the_same_key() -> None:
    assert query_cache_key("cilium", "kb__openai__m__1536__c1") == query_cache_key(
        "cilium", "kb__openai__m__1536__c1"
    )


def test_the_same_query_in_a_different_collection_is_a_different_key() -> None:
    # AD-021. The collection encodes provider, model, dimensions, and chunker
    # version; a shared key across two of them would hand back a vector from
    # the wrong space and the search would answer with plausible nonsense.
    assert query_cache_key("cilium", "kb__openai__m__1536__c1") != query_cache_key(
        "cilium", "kb__openai__m__512__c1"
    )


def test_the_separator_cannot_be_forged_from_the_inputs() -> None:
    # Without a separator, ("ab", "c") and ("a", "bc") collide.
    assert query_cache_key("ab", "c") != query_cache_key("a", "bc")


# --- hit, miss, and expiry -----------------------------------------------


def test_a_stored_vector_comes_back() -> None:
    instance, _ = cache()
    instance.put("k", entry())

    assert instance.get("k") == entry()


def test_an_unknown_key_misses() -> None:
    instance, _ = cache()

    assert instance.get("k") is None
    assert instance.stats.misses == 1


def test_an_entry_expires_after_its_ttl() -> None:
    instance, clock = cache(ttl_seconds=60.0)
    instance.put("k", entry())

    clock.advance(60.0)

    assert instance.get("k") is None


def test_an_entry_survives_right_up_to_its_ttl() -> None:
    instance, clock = cache(ttl_seconds=60.0)
    instance.put("k", entry())

    clock.advance(59.9)

    assert instance.get("k") is not None


def test_an_expired_entry_is_dropped_rather_than_left_in_place() -> None:
    # Expiry is lazy; a read is what reclaims the slot. If it did not, the
    # bound would still hold but the entry would be re-checked on every read.
    instance, clock = cache(ttl_seconds=60.0)
    instance.put("k", entry())
    clock.advance(61.0)

    instance.get("k")

    assert instance.stats.entries == 0


def test_a_refreshed_key_gets_a_new_deadline() -> None:
    instance, clock = cache(ttl_seconds=60.0)
    instance.put("k", entry())
    clock.advance(50.0)

    instance.put("k", entry(2.0))
    clock.advance(50.0)

    assert instance.get("k") == entry(2.0)


# --- the bound -----------------------------------------------------------


def test_the_cache_never_holds_more_than_its_capacity() -> None:
    instance, _ = cache(max_entries=3)

    for index in range(10):
        instance.put(f"k{index}", entry(float(index)))

    assert instance.stats.entries == 3


def test_the_least_recently_used_entry_is_the_one_evicted() -> None:
    instance, _ = cache(max_entries=2)
    instance.put("a", entry(1.0))
    instance.put("b", entry(2.0))

    instance.get("a")  # `a` is now the most recent, so `b` should go
    instance.put("c", entry(3.0))

    assert instance.get("a") is not None
    assert instance.get("b") is None
    assert instance.get("c") is not None


def test_a_zero_capacity_cache_stores_nothing() -> None:
    # An operator turning the cache off should not need the call sites to grow
    # a branch: every lookup simply misses.
    instance, _ = cache(max_entries=0)
    instance.put("k", entry())

    assert instance.get("k") is None
    assert instance.stats.entries == 0


def test_stats_count_hits_and_misses_from_the_start() -> None:
    instance, _ = cache()
    instance.put("k", entry())

    instance.get("k")
    instance.get("k")
    instance.get("missing")

    stats = instance.stats
    assert (stats.hits, stats.misses, stats.capacity) == (2, 1, 4)
