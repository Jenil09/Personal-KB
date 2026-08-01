"""`POST /v1/documents` end to end (PRD §6.3).

The full app through `httpx.ASGITransport` — real routers, real services, real
adapters, real Postgres, real Chroma — with the embedding provider stubbed at
the port, which is what Design §7 specifies for this layer. What is asserted
here is the HTTP contract: status codes, the response body's shape, auth, the
body cap, and that the lifespan actually wires everything together.

The flow's own guarantees are covered in `test_ingestion.py`. Repeating them
through HTTP would be slower and would prove the same thing twice.
"""

import hashlib
import itertools
from collections.abc import AsyncIterator, Sequence
from http import HTTPStatus

import httpx
import pytest
from testcontainers.community.chroma import ChromaContainer

from ai_embeddings import EmbeddingModel, EmbeddingProvider, RawEmbeddings
from ai_embeddings.port import Purpose, TokenSource
from kb_api.config import KbApiSettings
from kb_api.main import build_app

pytestmark = pytest.mark.integration

DIMENSIONS = 16
API_KEY = "s3cr3t"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
BODY_LIMIT = 32 * 1024

MODEL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=DIMENSIONS,
    max_input_tokens=8191,
    max_batch_inputs=96,
)

_RUNS = itertools.count()


class StubProvider(EmbeddingProvider):
    """Stubbed at the port, per Design §7 — nothing below it is faked."""

    @property
    def model(self) -> EmbeddingModel:
        return MODEL

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        assert purpose is Purpose.DOCUMENT
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
async def client(
    migrated: str, chroma: ChromaContainer, chroma_port: int
) -> AsyncIterator[httpx.AsyncClient]:
    settings = KbApiSettings(
        api_keys={"n8n": API_KEY},
        postgres={"dsn": migrated},
        chroma={"host": "127.0.0.1", "port": chroma_port},
        openai={"api_key": "unused-because-the-driver-is-stubbed"},
        # A per-run default so each test module gets a collection of its own;
        # Chroma has no truncate and the container is shared.
        default_provider=f"openai{next(_RUNS)}",
        max_body_bytes=BODY_LIMIT,
    )
    app = build_app(settings, providers={settings.default_provider: StubProvider()})
    # The lifespan is entered explicitly: it is what opens Chroma and runs the
    # startup reconciliation, and a test that skipped it would be testing an
    # app the deployment never runs.
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kb") as instance,
    ):
        yield instance


async def test_health_reports_both_dependencies(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["status"] == "ok"
    assert body["postgres"] == "connected"
    assert body["chromadb"] == "connected"


async def test_ingesting_a_document_returns_201_and_the_prd_body(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/v1/documents", json=document(), headers=AUTH)

    assert response.status_code == HTTPStatus.CREATED
    body = response.json()
    assert body["status"] == "success"
    assert body["chunks_created"] > 0
    assert body["chunks_reused"] == 0
    assert body["total_tokens"] > 0
    assert body["collection"].startswith("kb__openai")


async def test_an_identical_re_ingest_answers_unchanged_with_the_same_id(
    client: httpx.AsyncClient,
) -> None:
    first = (await client.post("/v1/documents", json=document(), headers=AUTH)).json()

    second = (await client.post("/v1/documents", json=document(), headers=AUTH)).json()

    assert second["status"] == "unchanged"
    assert second["document_id"] == first["document_id"]
    assert second["total_tokens"] == 0


async def test_an_edited_re_ingest_reports_what_it_replaced(client: httpx.AsyncClient) -> None:
    first = (await client.post("/v1/documents", json=document(), headers=AUTH)).json()

    edited = document(content=CONTENT.replace("## Section 2", "## Section 2 revised"))
    second = (await client.post("/v1/documents", json=edited, headers=AUTH)).json()

    assert second["superseded"] == [first["document_id"]]
    assert second["chunks_reused"] > 0


async def test_no_api_key_is_401(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/documents", json=document())

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["code"] == "unauthenticated"


async def test_a_wrong_api_key_is_the_same_401(client: httpx.AsyncClient) -> None:
    # AD-011: undifferentiated, so a rejection cannot be used to probe which
    # keys exist.
    response = await client.post(
        "/v1/documents", json=document(), headers={"Authorization": "Bearer wrong"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["code"] == "unauthenticated"


async def test_an_unknown_provider_is_422(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/documents", json=document(provider="cohere"), headers=AUTH)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["code"] == "validation_error"


async def test_a_missing_field_is_422_as_problem_json(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/documents", json={"title": "no content"}, headers=AUTH)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_an_oversized_body_is_413(client: httpx.AsyncClient) -> None:
    # Technical Design §8. The cap is enforced by the app, not only by Nginx.
    response = await client.post(
        "/v1/documents", json=document(content="x" * (BODY_LIMIT * 2)), headers=AUTH
    )

    assert response.status_code == HTTPStatus.CONTENT_TOO_LARGE
    assert response.json()["code"] == "payload_too_large"


async def test_every_response_carries_a_request_id(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/documents", json=document(), headers=AUTH)

    assert response.headers["X-Request-ID"]


async def test_the_openapi_schema_describes_the_endpoint(client: httpx.AsyncClient) -> None:
    # AD-016. Phase 8 commits the schema and drift-checks it; this only asserts
    # the endpoint is in it at all, with the auth requirement attached.
    schema = (await client.get("/openapi.json")).json()

    operation = schema["paths"]["/v1/documents"]["post"]
    assert operation["security"]
    assert "413" in operation["responses"]
