"""Document management end to end (PRD §6.4-6.6), plus the scope split (AD-024).

Same shape as `test_ingest_api.py`: the whole app through `httpx.ASGITransport`,
real routers, services, Postgres and Chroma, embedding stubbed at the port.

The tests worth reading twice are the two that are not about a status code — the
one that proves a delete removes the vectors and not just the row, and the one
that proves an `search`-scoped key cannot delete. The first is the reason the
delete flow is ordered the way it is; the second is the whole of AD-024.
"""

import hashlib
import itertools
from collections.abc import AsyncIterator, Sequence
from http import HTTPStatus
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text as sql
from testcontainers.community.chroma import ChromaContainer

from ai_embeddings import EmbeddingModel, EmbeddingProvider, RawEmbeddings
from ai_embeddings.port import Purpose, TokenSource
from kb_api.config import KbApiSettings
from kb_api.main import build_app
from platform_db import Database, DatabaseSettings

pytestmark = pytest.mark.integration

TRUNCATE = sql(
    "TRUNCATE kb.documents, kb.chunks, kb_audit.request_logs, "
    "kb_audit.token_usage_logs, kb_audit.ingest_logs, kb_audit.error_logs "
    "RESTART IDENTITY CASCADE"
)

DIMENSIONS = 16
WRITE_KEY = "cli-secret"
SEARCH_KEY = "n8n-secret"
WRITER = {"Authorization": f"Bearer {WRITE_KEY}"}
READER = {"Authorization": f"Bearer {SEARCH_KEY}"}

MODEL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=DIMENSIONS,
    max_input_tokens=8191,
    max_batch_inputs=96,
)

_RUNS = itertools.count()


class StubProvider(EmbeddingProvider):
    @property
    def model(self) -> EmbeddingModel:
        return MODEL

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(DIMENSIONS)]
            norm = sum(value * value for value in raw) ** 0.5
            vectors.append(tuple(value / norm for value in raw))
        return RawEmbeddings(
            tuple(vectors), sum(self.count_tokens(text) for text in texts), TokenSource.PROVIDER
        )


CONTENT = "\n\n".join(f"## Section {i}\n\n" + f"body{i} " * 200 for i in range(4))


def document(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "title": "Redshift Architecture",
        "content": CONTENT,
        "type": "architecture",
        "source": "redshift.md",
        "tags": ["ansible", "hardening"],
    }
    body.update(overrides)
    return body


@pytest.fixture
async def clean(migrated: str) -> AsyncIterator[None]:
    """An empty corpus per test.

    The Postgres container is module-scoped because starting one costs more than
    every test in the file put together, so rows would otherwise accumulate
    across tests — and most of the assertions here are counts. Truncating on the
    way in rather than the way out means a failed test leaves its rows behind to
    be looked at.
    """
    database = Database(DatabaseSettings(dsn=migrated, connect_timeout_seconds=5.0))
    try:
        async with database.engine.begin() as connection:
            await connection.execute(TRUNCATE)
        yield
    finally:
        await database.dispose()


@pytest.fixture
async def client(
    clean: None, migrated: str, chroma: ChromaContainer, chroma_port: int, tmp_path: Path
) -> AsyncIterator[httpx.AsyncClient]:
    settings = KbApiSettings(
        api_keys={"n8n": SEARCH_KEY, "cli": WRITE_KEY},
        # AD-024 as the deployment configures it: the workflow key reads, the
        # operator key does everything.
        api_key_scopes={"n8n": frozenset({"search"}), "cli": frozenset({"search", "write"})},
        postgres={"dsn": migrated},
        chroma={"host": "127.0.0.1", "port": chroma_port},
        openai={"api_key": "unused-because-the-driver-is-stubbed"},
        default_provider=f"openai{next(_RUNS)}",
        # Off the repo's working directory: the default is relative, and a test
        # run must not leave records in a path a deployment would use.
        audit={"spill_path": tmp_path / "audit.spill.jsonl"},
    )
    app = build_app(settings, providers={settings.default_provider: StubProvider()})
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kb") as instance,
    ):
        yield instance


async def _ingest(client: httpx.AsyncClient, **overrides: object) -> str:
    response = await client.post("/v1/documents", json=document(**overrides), headers=WRITER)
    assert response.status_code == HTTPStatus.CREATED, response.text
    return str(response.json()["document_id"])


# --- listing (PRD §6.4) ---------------------------------------------------


async def test_listing_returns_a_page_without_content(client: httpx.AsyncClient) -> None:
    await _ingest(client)

    response = await client.get("/v1/documents", headers=READER)

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["total"] == 1
    assert body["limit"] == 50
    entry = body["documents"][0]
    assert entry["title"] == "Redshift Architecture"
    assert entry["status"] == "indexed"
    assert entry["chunk_count"] > 0
    # The listing is a catalogue, not a bulk export: a fifty-item page carrying
    # content would be a 25 MB answer to "what is in here".
    assert "content" not in entry


async def test_listing_filters_by_type_and_tags(client: httpx.AsyncClient) -> None:
    await _ingest(client, source="a.md", type="architecture", tags=["ansible"])
    await _ingest(client, source="b.md", type="runbook", tags=["backup"], content=CONTENT + " b")

    by_type = (await client.get("/v1/documents?type=runbook", headers=READER)).json()
    by_tag = (await client.get("/v1/documents?tags=ansible", headers=READER)).json()

    assert by_type["total"] == 1
    assert by_type["documents"][0]["type"] == "runbook"
    assert by_tag["total"] == 1
    assert by_tag["documents"][0]["tags"] == ["ansible"]


async def test_listing_paginates_with_a_stable_order(client: httpx.AsyncClient) -> None:
    for index in range(3):
        await _ingest(client, source=f"{index}.md", content=f"{CONTENT} {index}")

    first = (await client.get("/v1/documents?limit=2&offset=0", headers=READER)).json()
    second = (await client.get("/v1/documents?limit=2&offset=2", headers=READER)).json()

    assert first["total"] == second["total"] == 3
    assert len(first["documents"]) == 2
    assert len(second["documents"]) == 1
    # The tiebreak on id is what stops a shared microsecond from dropping or
    # repeating a row across pages.
    ids = [entry["id"] for entry in first["documents"] + second["documents"]]
    assert len(set(ids)) == 3


async def test_an_unknown_filter_value_returns_an_empty_page_not_an_error(
    client: httpx.AsyncClient,
) -> None:
    await _ingest(client)

    body = (await client.get("/v1/documents?type=nonexistent", headers=READER)).json()

    assert body == {"documents": [], "total": 0, "limit": 50, "offset": 0}


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
async def test_out_of_range_pagination_is_422(client: httpx.AsyncClient, query: str) -> None:
    response = await client.get(f"/v1/documents?{query}", headers=READER)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


# --- single get (PRD §6.5) ------------------------------------------------


async def test_getting_one_document_includes_content_and_provenance(
    client: httpx.AsyncClient,
) -> None:
    document_id = await _ingest(client)

    response = await client.get(f"/v1/documents/{document_id}", headers=READER)

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["id"] == document_id
    assert body["content"].startswith("## Section 0")
    # AD-014's provenance, recorded from the transport rather than the body.
    assert body["ingested_by_key_id"] == "cli"


async def test_getting_a_missing_document_is_404(client: httpx.AsyncClient) -> None:
    response = await client.get(f"/v1/documents/{uuid4()}", headers=READER)

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["code"] == "not_found"


async def test_a_malformed_id_is_422_not_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/documents/not-a-uuid", headers=READER)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


# --- delete (PRD §6.6) ----------------------------------------------------


async def test_deleting_removes_the_document_from_both_stores(
    client: httpx.AsyncClient,
) -> None:
    """The one that matters: a delete that dropped the row and left the vectors
    would pass every status-code assertion above and still answer searches with
    a document the caller believes is gone."""
    document_id = await _ingest(client)
    before = (await client.post("/v1/search", json={"query": "body1"}, headers=READER)).json()
    assert before["results"]

    response = await client.delete(f"/v1/documents/{document_id}", headers=WRITER)

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert (await client.get(f"/v1/documents/{document_id}", headers=READER)).status_code == 404
    assert (await client.get("/v1/documents", headers=READER)).json()["total"] == 0
    after = await client.post("/v1/search", json={"query": "body1"}, headers=READER)
    # The collection still exists but holds nothing, which search reports as a
    # 409 rather than an empty result set (Design §3.2).
    assert after.status_code in (HTTPStatus.OK, HTTPStatus.CONFLICT)
    if after.status_code == HTTPStatus.OK:
        assert after.json()["results"] == []


async def test_deleting_is_idempotent(client: httpx.AsyncClient) -> None:
    """A client that timed out mid-delete cannot tell which side of the purge it
    stopped on, so the retry has to be safe."""
    document_id = await _ingest(client)

    first = await client.delete(f"/v1/documents/{document_id}", headers=WRITER)
    second = await client.delete(f"/v1/documents/{document_id}", headers=WRITER)

    assert first.status_code == second.status_code == HTTPStatus.NO_CONTENT
    assert first.headers["X-Deleted"] == "true"
    assert second.headers["X-Deleted"] == "false"


async def test_deleting_something_that_never_existed_is_204(client: httpx.AsyncClient) -> None:
    response = await client.delete(f"/v1/documents/{uuid4()}", headers=WRITER)

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.headers["X-Deleted"] == "false"


async def test_deleting_leaves_other_documents_alone(client: httpx.AsyncClient) -> None:
    keep = await _ingest(client, source="keep.md")
    remove = await _ingest(client, source="remove.md", content=CONTENT + " other")

    await client.delete(f"/v1/documents/{remove}", headers=WRITER)

    listing = (await client.get("/v1/documents", headers=READER)).json()
    assert [entry["id"] for entry in listing["documents"]] == [keep]


# --- scopes (AD-024) ------------------------------------------------------


async def test_a_search_scoped_key_cannot_delete(client: httpx.AsyncClient) -> None:
    """The reason AD-024 exists. n8n processes scraped, untrusted content; a
    prompt injection that reaches its key must not be able to purge the corpus."""
    document_id = await _ingest(client)

    response = await client.delete(f"/v1/documents/{document_id}", headers=READER)

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json()["code"] == "insufficient_scope"
    assert response.json()["required_scope"] == "write"
    # And the document is still there.
    assert (await client.get(f"/v1/documents/{document_id}", headers=READER)).status_code == 200


async def test_a_search_scoped_key_cannot_ingest(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/documents", json=document(), headers=READER)

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_a_search_scoped_key_can_search_and_read(client: httpx.AsyncClient) -> None:
    document_id = await _ingest(client)

    assert (await client.get("/v1/documents", headers=READER)).status_code == 200
    assert (await client.get(f"/v1/documents/{document_id}", headers=READER)).status_code == 200
    assert (
        await client.post("/v1/search", json={"query": "body0"}, headers=READER)
    ).status_code == 200


async def test_a_search_scoped_key_cannot_read_admin_stats(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/admin/stats", headers=READER)

    assert response.status_code == HTTPStatus.FORBIDDEN


# --- admin stats (PRD §6.7) -----------------------------------------------


async def test_stats_report_the_corpus_and_the_index_side_by_side(
    client: httpx.AsyncClient,
) -> None:
    await _ingest(client)

    body = (await client.get("/v1/admin/stats", headers=WRITER)).json()

    assert body["total_documents"] == 1
    assert body["documents_by_status"] == {"indexed": 1}
    assert body["total_chunks"] > 0
    assert body["total_tokens_stored"] > 0
    collection = body["collections"][0]
    assert collection["dimensions"] == DIMENSIONS
    assert collection["vectors"] == body["total_chunks"]


async def test_stats_report_observability_health(client: httpx.AsyncClient) -> None:
    """The part that justifies the endpoint: a dropped telemetry record and a
    spilled audit record are both invisible unless something asks."""
    body = (await client.get("/v1/admin/stats", headers=WRITER)).json()

    assert body["audit_spill_depth"] == 0
    assert body["telemetry_dropped"] == 0
    assert body["recent_bursts"] == 0


async def test_health_reports_the_audit_surfaces_too(client: httpx.AsyncClient) -> None:
    body = (await client.get("/health")).json()

    assert body["status"] == "ok"
    assert body["postgres"] == "connected"
    assert body["chromadb"] == "connected"
    assert body["audit_spill"] == "empty"
    assert body["telemetry_queue"].startswith("0/")
