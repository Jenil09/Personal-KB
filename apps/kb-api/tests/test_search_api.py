"""The Phase 7 exit criteria, and `POST /v1/search` end to end (PRD §6.2).

The full app through `httpx.ASGITransport` — real routers, real services, real
Chroma, real Postgres — with only the embedding provider stubbed, at the port
(Design §7). Every ranking assertion here is therefore a claim about what Chroma
actually returned for a real vector query, not about what a fake was told to say.

**The stub embeds rather than pretends to.** It is a hashing bag-of-words
vectoriser: tokens are hashed into buckets, counted, and the vector is
L2-normalised. That makes cosine similarity behave like lexical overlap, which
is weaker than a real embedding model and, for these tests, better — the
expected ranking is derivable from the text without a model in the loop, so an
assertion about which chunk comes first is checkable by reading the fixture.

The three exit criteria live in their own sections below: p95 latency, tag
filter parity against a naive in-memory filter, and the `409` for a provider
whose collection holds nothing.
"""

import asyncio
import itertools
import math
import re
from collections.abc import AsyncIterator, Sequence
from http import HTTPStatus
from pathlib import Path
from statistics import quantiles

import httpx
import pytest
from testcontainers.community.chroma import ChromaContainer

from ai_embeddings import EmbeddingModel, EmbeddingProvider, RawEmbeddings
from ai_embeddings.port import Purpose, TokenSource
from kb_api.config import KbApiSettings
from kb_api.main import build_app

pytestmark = pytest.mark.integration

DIMENSIONS = 64
API_KEY = "s3cr3t"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

# Design §3.2 budgets 200-400 ms for the query embedding, and calls it the
# dominant term. The stub waits the top of that range on every uncached call so
# the p95 measured below includes the cost the real deployment pays.
EMBED_LATENCY_SECONDS = 0.4

MODEL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=DIMENSIONS,
    max_input_tokens=8191,
    max_batch_inputs=96,
)

_RUNS = itertools.count()
_TOKEN = re.compile(r"[a-z0-9]+")

AI_KB = Path(__file__).resolve().parents[3] / "ai-kb"


class StubProvider(EmbeddingProvider):
    """A hashing vectoriser, so similarity tracks lexical overlap.

    `latency` is the simulated provider round trip. It is applied per call and
    not on a cache hit, which is what makes the cached row of §3.2's latency
    table measurable rather than asserted from the code's shape.
    """

    def __init__(self, *, latency: float = 0.0) -> None:
        self.latency = latency
        self.calls = 0
        self.queries: list[str] = []

    @property
    def model(self) -> EmbeddingModel:
        return MODEL

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        self.calls += 1
        if purpose is Purpose.QUERY:
            self.queries.extend(texts)
        if self.latency:
            await asyncio.sleep(self.latency)
        return RawEmbeddings(
            tuple(_vectorise(text) for text in texts),
            sum(self.count_tokens(text) for text in texts),
            TokenSource.PROVIDER,
        )


def _vectorise(text: str) -> tuple[float, ...]:
    """Bag of hashed tokens, L2-normalised so cosine distance is meaningful."""
    buckets = [0.0] * DIMENSIONS
    for token in _TOKEN.findall(text.lower()):
        buckets[hash_token(token)] += 1.0
    norm = math.sqrt(sum(value * value for value in buckets))
    if norm == 0.0:
        # Chroma's cosine space cannot place a zero vector, and a chunk of pure
        # punctuation is the only way to get one.
        return tuple([1.0] + [0.0] * (DIMENSIONS - 1))
    return tuple(value / norm for value in buckets)


def hash_token(token: str) -> int:
    """A stable bucket. `hash()` is salted per process and would not be."""
    digest = 0
    for character in token:
        digest = (digest * 31 + ord(character)) % 1_000_003
    return digest % DIMENSIONS


def _settings(*, migrated: str, chroma_port: int, providers: Sequence[str]) -> KbApiSettings:
    return KbApiSettings(
        api_keys={"n8n": API_KEY},
        postgres={"dsn": migrated},
        chroma={"host": "127.0.0.1", "port": chroma_port},
        openai={"api_key": "unused-because-the-driver-is-stubbed"},
        # A per-test collection: Chroma has no truncate and the container is
        # shared, so a corpus left behind would show up as extra results here.
        default_provider=providers[0],
        query_cache_size=64,
    )


@pytest.fixture
def provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def names() -> tuple[str, str]:
    """A populated provider and an unpopulated one, unique to this test."""
    run = next(_RUNS)
    return f"openai{run}", f"gemini{run}"


@pytest.fixture
async def client(
    migrated: str,
    chroma: ChromaContainer,
    chroma_port: int,
    provider: StubProvider,
    names: tuple[str, str],
) -> AsyncIterator[httpx.AsyncClient]:
    settings = _settings(migrated=migrated, chroma_port=chroma_port, providers=names)
    # Two providers, one of which nothing is ever ingested into — that is the
    # `409` case, and configuring it here rather than in one test keeps the
    # unpopulated collection genuinely unpopulated.
    app = build_app(settings, providers={names[0]: provider, names[1]: StubProvider()})
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kb") as instance,
    ):
        yield instance


async def ingest(client: httpx.AsyncClient, **body: object) -> str:
    response = await client.post("/v1/documents", json=body, headers=AUTH)
    assert response.status_code == HTTPStatus.CREATED, response.text
    return str(response.json()["document_id"])


async def search(client: httpx.AsyncClient, **body: object) -> httpx.Response:
    return await client.post("/v1/search", json=body, headers=AUTH)


def section(heading: str, *sentences: str) -> str:
    """A chunk-sized section: long enough that the chunker does not merge it."""
    filler = " ".join(sentences)
    return f"## {heading}\n\n{filler} " + f"{heading.lower()} detail. " * 40


@pytest.fixture
async def corpus(client: httpx.AsyncClient) -> dict[str, str]:
    """Six short, distinctly worded documents with overlapping tags."""
    documents = {
        "networking": (
            "Networking",
            section("Cilium", "Bare metal kubernetes with cilium in kube-proxy replacement mode."),
            "architecture",
            ["ansible", "hardening"],
        ),
        "storage": (
            "Storage",
            section("Longhorn", "Longhorn replicates every volume three times across nvme."),
            "architecture",
            ["hardening"],
        ),
        "backups": (
            "Backups",
            section("Restic", "Restic ships nightly snapshots to object storage offsite."),
            "runbook",
            ["ansible"],
        ),
        "provisioning": (
            "Provisioning",
            section("Terraform", "Terraform provisions the hosts and ansible configures them."),
            "runbook",
            ["ansible", "provisioning"],
        ),
        "observability": (
            "Observability",
            section("Grafana", "Grafana dashboards read prometheus metrics scraped per node."),
            "runbook",
            [],
        ),
        "identity": (
            "Identity",
            section("Authelia", "Authelia fronts every service with single sign on."),
            "architecture",
            ["hardening", "provisioning"],
        ),
    }
    identifiers = {}
    for name, (title, content, kind, tags) in documents.items():
        identifiers[name] = await ingest(
            client,
            title=title,
            content=content,
            type=kind,
            source=f"{name}.md",
            tags=tags,
        )
    return identifiers


# --- the contract, PRD §6.2 ----------------------------------------------


async def test_a_search_returns_the_prd_response_body(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    response = await search(client, query="cilium kube-proxy replacement", top_k=3)

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert len(body["results"]) == 3
    assert body["query_tokens"] > 0
    assert body["latency_ms"] >= 0
    first = body["results"][0]
    assert set(first) == {"id", "text", "metadata", "score"}
    assert set(first["metadata"]) == {"document_id", "title", "type", "tags", "source", "ordinal"}


async def test_the_best_match_is_the_document_the_query_describes(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # A real Chroma query over real vectors. The stub makes similarity lexical,
    # so the expected answer is readable off the fixture.
    response = await search(client, query="longhorn replicates volumes across nvme", top_k=1)

    assert response.json()["results"][0]["metadata"]["document_id"] == corpus["storage"]


async def test_results_come_back_ranked_by_descending_score(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    body = (await search(client, query="ansible terraform provisioning", top_k=6)).json()

    scores = [result["score"] for result in body["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_tags_come_back_as_a_list_not_the_stored_delimited_string(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # AD-005 stores tags pipe-delimited because Chroma rejects lists. That is a
    # storage detail and must not reach the caller.
    body = (await search(client, query="cilium kube-proxy", top_k=1)).json()

    assert body["results"][0]["metadata"]["tags"] == ["ansible", "hardening"]


async def test_a_scalar_filter_restricts_the_result_set(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    body = (await search(client, query="kubernetes", top_k=10, filters={"type": "runbook"})).json()

    assert body["results"]
    assert {result["metadata"]["type"] for result in body["results"]} == {"runbook"}


async def test_top_k_bounds_the_result_set(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    body = (await search(client, query="kubernetes", top_k=2)).json()

    assert len(body["results"]) == 2


# --- exit criterion: tag-filtered results match a naive in-memory filter --


async def naive_tag_filter(client: httpx.AsyncClient, query: str, tags: set[str]) -> list[str]:
    """Every result, ranked, then filtered in Python by the tags it carries.

    Deliberately the dumbest possible implementation of the same question. If
    the AD-005 path — Postgres ID lookup, `$in` clause, Chroma `where` — ever
    disagrees with this, the pipeline is filtering on something other than what
    the caller asked for.
    """
    body = (await search(client, query=query, top_k=50)).json()
    return [result["id"] for result in body["results"] if tags & set(result["metadata"]["tags"])]


@pytest.mark.parametrize(
    "tags", [{"ansible"}, {"hardening"}, {"provisioning"}, {"ansible", "provisioning"}]
)
async def test_a_tag_filter_matches_the_naive_filter_exactly(
    client: httpx.AsyncClient, corpus: dict[str, str], tags: set[str]
) -> None:
    query = "kubernetes storage provisioning and identity"

    filtered = (await search(client, query=query, top_k=50, filters={"tags": sorted(tags)})).json()

    assert [result["id"] for result in filtered["results"]] == await naive_tag_filter(
        client, query, tags
    )


async def test_match_all_tags_is_the_intersection_not_the_union(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # Only `provisioning.md` carries both.
    body = (
        await search(
            client,
            query="kubernetes",
            top_k=50,
            filters={"tags": ["ansible", "provisioning"], "match_all_tags": True},
        )
    ).json()

    assert {result["metadata"]["document_id"] for result in body["results"]} == {
        corpus["provisioning"]
    }


async def test_a_tag_and_a_type_filter_apply_together(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # The `$and` case: one term from Postgres, one from Chroma metadata. The
    # pair has to discriminate for the test to mean anything — `ansible` alone
    # would also match `networking.md`, and `runbook` alone `observability.md`.
    body = (
        await search(
            client,
            query="kubernetes",
            top_k=50,
            filters={"tags": ["ansible"], "type": "runbook"},
        )
    ).json()

    assert {result["metadata"]["document_id"] for result in body["results"]} == {
        corpus["backups"],
        corpus["provisioning"],
    }


async def test_a_tag_nothing_carries_returns_an_empty_result_set(
    client: httpx.AsyncClient, corpus: dict[str, str], provider: StubProvider
) -> None:
    provider.queries.clear()

    body = (await search(client, query="kubernetes", filters={"tags": ["nonexistent"]})).json()

    assert body["results"] == []
    # Nothing could have matched whatever the query vector was, so nothing was
    # sent to the provider — and `query_tokens` says so.
    assert provider.queries == []
    assert body["query_tokens"] == 0


async def test_a_deleted_documents_chunks_stop_being_tag_matchable(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # Superseding is the only delete Phase 6 ships (AD-020). The replaced
    # document's IDs must leave the pre-filter with its rows.
    await ingest(
        client,
        title="Storage",
        content=section("Longhorn", "Longhorn now replicates volumes four times."),
        type="architecture",
        source="storage.md",
        tags=["hardening"],
    )

    body = (
        await search(client, query="longhorn", top_k=50, filters={"tags": ["hardening"]})
    ).json()

    assert corpus["storage"] not in {
        result["metadata"]["document_id"] for result in body["results"]
    }


# --- exit criterion: an unpopulated collection is a 409 ------------------


async def test_a_provider_with_an_empty_collection_is_409(
    client: httpx.AsyncClient, corpus: dict[str, str], names: tuple[str, str]
) -> None:
    response = await search(client, query="cilium", provider=names[1])

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["code"] == "conflict"


async def test_the_409_costs_no_embedding_call(
    client: httpx.AsyncClient,
    corpus: dict[str, str],
    names: tuple[str, str],
    provider: StubProvider,
) -> None:
    before = provider.calls

    await search(client, query="cilium", provider=names[1])

    assert provider.calls == before


async def test_an_unknown_provider_is_422_not_409(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # A missing API key and an empty corpus are different problems with
    # different fixes; answering them the same way hides one behind the other.
    response = await search(client, query="cilium", provider="cohere")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["code"] == "validation_error"


# --- auth, validation, and the schema ------------------------------------


async def test_search_without_an_api_key_is_401(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/search", json={"query": "cilium"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_an_empty_query_is_422(client: httpx.AsyncClient, corpus: dict[str, str]) -> None:
    response = await search(client, query="")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_a_top_k_over_the_ceiling_is_422(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    response = await search(client, query="cilium", top_k=500)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_a_misspelled_filter_key_is_422_rather_than_silently_ignored(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    # The one validation failure that would otherwise be invisible: the search
    # succeeds, the filter does not apply, and the larger result set looks
    # entirely plausible.
    response = await search(client, query="cilium", filters={"tag": ["ansible"]})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_the_openapi_schema_describes_the_endpoint(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()

    operation = schema["paths"]["/v1/search"]["post"]
    assert operation["security"]
    assert "409" in operation["responses"]


# --- exit criterion: p95 under 1.5 s -------------------------------------


async def realistic_corpus(client: httpx.AsyncClient) -> int:
    """The real `ai-kb/` documents — the corpus this service exists to search.

    ~75 chunks across four documents. Small against the 6,700-chunk projection
    in Design §8, but Chroma's query cost is flat at both sizes (4.7 ms
    measured over the full index in Phase 6), and these are the actual files
    rather than generated filler.
    """
    total = 0
    for path in sorted(AI_KB.glob("*.md")):
        response = await client.post(
            "/v1/documents",
            json={
                "title": path.stem,
                "content": path.read_text(encoding="utf-8"),
                "type": "design",
                "source": path.name,
                "tags": ["design"],
            },
            headers=AUTH,
        )
        assert response.status_code == HTTPStatus.CREATED, response.text
        total += int(response.json()["chunks_created"])
    return total


async def test_p95_latency_is_within_the_prd_budget(
    client: httpx.AsyncClient, provider: StubProvider
) -> None:
    """PRD §11: p95 < 1.5 s. Measured, with the provider round trip included.

    The embedding call is stubbed, so its latency is simulated rather than
    real — at 400 ms, the *top* of the 200-400 ms range Design §3.2 budgets for
    it. Everything else in the measurement is genuine: real chunk vectors in a
    real Chroma, the real HNSW query, real serialisation, the real middleware
    stack. What this cannot cover is the variance of OpenAI's own tail, which
    is why the budget has a gigabyte of headroom over the measured figure.
    """
    assert await realistic_corpus(client) > 50
    provider.latency = EMBED_LATENCY_SECONDS

    # Distinct queries, so every one of them misses the cache and pays the
    # embedding round trip. Design §8 says that is the normal path for this
    # workload, not the exception.
    latencies = []
    for index in range(20):
        response = await search(client, query=f"how is chroma deployed variant {index}", top_k=5)
        assert response.status_code == HTTPStatus.OK
        latencies.append(response.json()["latency_ms"] / 1000)

    p95 = quantiles(latencies, n=20)[-1]
    assert p95 < 1.5, f"p95 {p95:.3f}s over the 1.5 s budget: {latencies}"


async def test_a_cached_query_skips_the_provider_round_trip(
    client: httpx.AsyncClient, corpus: dict[str, str], provider: StubProvider
) -> None:
    """§3.2's cached row. Design §8 calls it an uncommon path, not the normal one."""
    provider.latency = EMBED_LATENCY_SECONDS

    cold = (await search(client, query="cilium kube-proxy replacement mode")).json()
    warm = (await search(client, query="cilium kube-proxy replacement mode")).json()

    assert cold["latency_ms"] >= EMBED_LATENCY_SECONDS * 1000
    assert warm["latency_ms"] < cold["latency_ms"]
