"""The Phase 6 exit criteria, against real Postgres and real Chroma.

Only the embedding provider is stubbed, and only at the port (Design §7) — it is
the one dependency that costs money and a network round trip. Everything else
here is the real repository, the real Chroma adapter, and the real flow, because
the properties being asserted are properties of the interaction between the two
stores and a fake of either would assert nothing.

Every claim about *how much* was embedded is read off a counter on the stub, not
inferred from a response field the flow could compute wrongly. `chunks_reused`
and `embedding_calls` are checked against each other for exactly that reason: if
the flow lied about one, the counter catches it.
"""

import hashlib
import itertools
from collections.abc import Sequence

import pytest
from testcontainers.community.chroma import ChromaContainer

from ai_embeddings import EmbeddingModel, EmbeddingProvider, RawEmbeddings
from ai_embeddings.port import Purpose, TokenSource
from kb_api.adapters.chroma import ChromaVectorStore, create_chroma_client
from kb_api.adapters.postgres import ChunkRepository, DocumentRepository
from kb_api.chunking import CHUNKER_VERSION
from kb_api.domain import DocumentStatus
from kb_api.services import (
    IngestionService,
    IngestOutcome,
    IngestRequest,
    ProviderRegistry,
    ReconciliationService,
)
from platform_core import ValidationError
from platform_db import Database

pytestmark = pytest.mark.integration

DIMENSIONS = 32

MODEL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=DIMENSIONS,
    max_input_tokens=8191,
    max_batch_inputs=4,  # small, so batching is actually exercised
)


class CountingProvider(EmbeddingProvider):
    """A deterministic stand-in that records exactly what it was asked to embed.

    Vectors are a hash of the text, so the same text always produces the same
    vector — which means a carried-forward vector and a re-embedded one are
    indistinguishable by value. That is deliberate: it stops a test from passing
    because the vectors happened to differ, and forces the reuse assertions onto
    the call counter, which is what the exit criteria actually name.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.embedded: list[str] = []

    @property
    def model(self) -> EmbeddingModel:
        return MODEL

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        assert purpose is Purpose.DOCUMENT
        self.calls += 1
        self.embedded.extend(texts)
        return RawEmbeddings(
            tuple(_vector(text) for text in texts),
            sum(self.count_tokens(text) for text in texts),
            TokenSource.PROVIDER,
        )

    def reset(self) -> None:
        self.calls = 0
        self.embedded.clear()


def _vector(text: str) -> tuple[float, ...]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [(digest[index % len(digest)] / 255.0) - 0.5 for index in range(DIMENSIONS)]
    norm = sum(value * value for value in raw) ** 0.5
    return tuple(value / norm for value in raw)


def document_of(sections: int, *, edit: int | None = None) -> str:
    """A document that chunks to roughly one chunk per section.

    Each section is well past `min_tokens`, so the chunker does not merge them
    and `edit=n` changes exactly one chunk's text — which is what makes "editing
    one paragraph embeds only the affected chunks" a checkable claim rather than
    a hopeful one.
    """
    body = []
    for index in range(sections):
        marker = "EDITED " if index == edit else ""
        body.append(f"## Section {index}\n\n{marker}" + f"content{index} " * 400)
    return "\n\n".join(body)


@pytest.fixture
def provider() -> CountingProvider:
    return CountingProvider()


_COLLECTIONS = itertools.count()


@pytest.fixture
def registry(provider: CountingProvider) -> ProviderRegistry:
    """A registry whose collection name is unique to this test.

    Postgres is truncated between tests by the `database` fixture; Chroma has no
    truncate and the container is shared across the module. A collection left
    populated by one test would show up as carried-forward chunks in the next,
    which is precisely the number these tests exist to measure. Registering the
    driver under a per-test name gives each one a virgin collection, since the
    name is what the collection is derived from (Design §2.3).
    """
    name = f"openai{next(_COLLECTIONS)}"
    return ProviderRegistry({name: provider}, default=name, chunker_version=CHUNKER_VERSION)


@pytest.fixture
async def vectors(chroma: ChromaContainer, chroma_port: int) -> ChromaVectorStore:
    store = ChromaVectorStore(
        lambda: create_chroma_client(
            host="127.0.0.1",
            port=chroma_port,
            tenant="default_tenant",
            database="default_database",
        )
    )
    await store.connect()
    return store


@pytest.fixture
def ingestion(
    database: Database, vectors: ChromaVectorStore, registry: ProviderRegistry
) -> IngestionService:
    return IngestionService(
        sessions=database,
        documents=DocumentRepository(),
        chunks=ChunkRepository(),
        vectors=vectors,
        providers=registry,
    )


def request_of(content: str, *, source: str | None = "notes.md") -> IngestRequest:
    return IngestRequest(
        title="Redshift Architecture",
        content=content,
        type="architecture",
        source=source,
        tags=("ansible", "hardening"),
        ingested_by_key_id="test-key",
    )


# --- exit criterion: re-ingesting identical content makes zero embedding calls


async def test_a_first_ingest_embeds_every_chunk(
    ingestion: IngestionService, provider: CountingProvider
) -> None:
    result = await ingestion.ingest(request_of(document_of(6)))

    assert result.outcome is IngestOutcome.SUCCESS
    assert result.chunks_created == len(provider.embedded)
    assert result.chunks_reused == 0


async def test_re_ingesting_identical_content_embeds_nothing(
    ingestion: IngestionService, provider: CountingProvider
) -> None:
    content = document_of(6)
    first = await ingestion.ingest(request_of(content))
    provider.reset()

    second = await ingestion.ingest(request_of(content))

    assert second.outcome is IngestOutcome.UNCHANGED
    assert second.document_id == first.document_id
    assert provider.calls == 0
    assert provider.embedded == []


# --- exit criterion: editing one paragraph embeds only the affected chunks


async def test_editing_one_section_embeds_only_that_section(
    ingestion: IngestionService, provider: CountingProvider
) -> None:
    original = await ingestion.ingest(request_of(document_of(8)))
    assert original.chunks_created >= 8
    provider.reset()

    edited = await ingestion.ingest(request_of(document_of(8, edit=3)))

    # Read off the counter, not off the response — the response is what is
    # under test. The edit touches the chunk holding section 3 and the one
    # carrying its overlap tail, so a small number rather than exactly one.
    assert len(provider.embedded) < original.chunks_created / 2
    assert all("EDITED" in text or "content3" in text for text in provider.embedded)
    assert edited.chunks_reused == edited.chunks_created + edited.chunks_reused - len(
        provider.embedded
    )


async def test_a_carried_forward_chunk_gets_its_own_vector_not_a_shared_id(
    ingestion: IngestionService, vectors: ChromaVectorStore, registry: ProviderRegistry
) -> None:
    """AD-019. The reused chunk must be reachable under the *new* document.

    Sharing the source chunk's `chroma_id` would leave the vector's metadata
    naming the old document, so a search hit would be attributed to a document
    that no longer holds that text — and purging either document would take the
    other's vector with it.
    """
    await ingestion.ingest(request_of(document_of(8), source="first.md"))
    second = await ingestion.ingest(request_of(document_of(8, edit=7), source="second.md"))

    matches = await vectors.query(
        registry.default.collection,
        _vector("content0 " * 10),
        top_k=50,
        where={"document_id": str(second.document_id)},
    )

    assert len(matches) == second.chunks_created + second.chunks_reused


# --- supersede (AD-020)


async def test_re_ingesting_an_edited_file_replaces_the_previous_version(
    ingestion: IngestionService, database: Database
) -> None:
    first = await ingestion.ingest(request_of(document_of(6), source="notes.md"))

    second = await ingestion.ingest(request_of(document_of(6, edit=2), source="notes.md"))

    assert second.superseded == (first.document_id,)
    documents = DocumentRepository()
    async with database.session() as session:
        assert await documents.get(session, first.document_id, include_deleted=True) is None
        survivor = await documents.get(session, second.document_id)
    assert survivor is not None
    assert survivor.status is DocumentStatus.INDEXED


async def test_the_superseded_documents_vectors_are_gone(
    ingestion: IngestionService, vectors: ChromaVectorStore, registry: ProviderRegistry
) -> None:
    # The one that matters. Search reads Chroma alone (AD-004), so a vector
    # left behind is a result for a document nothing else admits exists.
    first = await ingestion.ingest(request_of(document_of(6), source="notes.md"))

    await ingestion.ingest(request_of(document_of(6, edit=2), source="notes.md"))

    stale = await vectors.query(
        registry.default.collection,
        _vector("content0 " * 10),
        top_k=50,
        where={"document_id": str(first.document_id)},
    )
    assert stale == ()


async def test_a_document_without_a_source_supersedes_nothing(
    ingestion: IngestionService,
) -> None:
    # There is nothing to match on. Two sourceless documents are two documents,
    # which is the honest answer when the caller gave no identity to key off.
    first = await ingestion.ingest(request_of(document_of(4), source=None))
    second = await ingestion.ingest(request_of(document_of(4, edit=1), source=None))

    assert second.superseded == ()
    assert second.document_id != first.document_id


# --- the flow's own guarantees


async def test_the_document_ends_indexed_with_its_chunk_count(
    ingestion: IngestionService, database: Database
) -> None:
    result = await ingestion.ingest(request_of(document_of(5)))

    async with database.session() as session:
        document = await DocumentRepository().get(session, result.document_id)
        rows = await ChunkRepository().for_document(session, result.document_id)

    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert document.chunk_count == len(rows)
    assert document.ingested_by_key_id == "test-key"


async def test_every_chunk_reaches_chroma(
    ingestion: IngestionService,
    database: Database,
    vectors: ChromaVectorStore,
    registry: ProviderRegistry,
) -> None:
    result = await ingestion.ingest(request_of(document_of(7)))

    async with database.session() as session:
        rows = await ChunkRepository().for_document(session, result.document_id)
    found = await vectors.fetch_vectors(
        registry.default.collection, [row.chroma_id for row in rows]
    )

    assert set(found) == {row.chroma_id for row in rows}


async def test_a_document_larger_than_one_batch_is_split_across_calls(
    ingestion: IngestionService, provider: CountingProvider
) -> None:
    # `max_batch_inputs` is 4 here. Without the split the port rejects the
    # batch and a large document becomes a 422 instead of two requests.
    result = await ingestion.ingest(request_of(document_of(10)))

    assert result.chunks_created > MODEL.max_batch_inputs
    assert provider.calls > 1
    assert result.embedding_calls == provider.calls


async def test_empty_content_is_rejected_before_anything_is_embedded(
    ingestion: IngestionService, provider: CountingProvider
) -> None:
    with pytest.raises(ValidationError):
        await ingestion.ingest(request_of("   \n\n  "))

    assert provider.calls == 0


# --- exit criterion: a crash between the Postgres commit and the Chroma upsert
#     self-heals on restart


async def test_a_crash_before_the_chroma_upsert_is_repaired_at_startup(
    ingestion: IngestionService,
    database: Database,
    vectors: ChromaVectorStore,
    registry: ProviderRegistry,
    provider: CountingProvider,
) -> None:
    """Injected, not simulated: the upsert is made to fail for real.

    What survives is exactly what §3.1 says survives — a committed document at
    `status = 'pending'` with its chunk rows and no vectors — and the assertion
    is that the startup pass turns that into a fully indexed document without
    being told which one to fix.
    """

    async def refuse(*_: object, **__: object) -> None:
        raise RuntimeError("chroma died mid-upsert")

    real_upsert = vectors.upsert
    vectors.upsert = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await ingestion.ingest(request_of(document_of(5)))
    vectors.upsert = real_upsert  # type: ignore[method-assign]

    async with database.session() as session:
        (stranded,) = await DocumentRepository().find_pending(session)
        rows = await ChunkRepository().for_document(session, stranded.id)
    assert rows, "the crash must leave the chunk rows the repair reads"
    assert await vectors.fetch_vectors(registry.default.collection, [rows[0].chroma_id]) == {}

    report = await ReconciliationService(
        sessions=database,
        documents=DocumentRepository(),
        chunks=ChunkRepository(),
        vectors=vectors,
        providers=registry,
    ).run()

    assert report.reindexed == 1
    assert report.failed == 0
    async with database.session() as session:
        repaired = await DocumentRepository().get(session, stranded.id)
    assert repaired is not None
    assert repaired.status is DocumentStatus.INDEXED
    found = await vectors.fetch_vectors(
        registry.default.collection, [row.chroma_id for row in rows]
    )
    assert set(found) == {row.chroma_id for row in rows}


async def test_a_crash_after_the_upsert_re_embeds_nothing(
    ingestion: IngestionService,
    database: Database,
    vectors: ChromaVectorStore,
    registry: ProviderRegistry,
    provider: CountingProvider,
) -> None:
    """The likely crash, and it must be free.

    The upsert precedes the status flip, so the window a restart most often
    lands in is one where every vector is already present. Re-embedding there
    would make a restart cost the price of the corpus.
    """
    result = await ingestion.ingest(request_of(document_of(5)))
    async with database.session() as session:
        await DocumentRepository().set_status(session, result.document_id, DocumentStatus.PENDING)
    provider.reset()

    report = await ReconciliationService(
        sessions=database,
        documents=DocumentRepository(),
        chunks=ChunkRepository(),
        vectors=vectors,
        providers=registry,
    ).run()

    assert report.reindexed == 1
    assert provider.calls == 0


async def test_a_crash_mid_supersede_purges_the_stale_vectors_at_startup(
    ingestion: IngestionService,
    database: Database,
    vectors: ChromaVectorStore,
    registry: ProviderRegistry,
) -> None:
    """The other window, and the worse one.

    A soft-deleted document whose vectors are still in the index is answerable
    by search and invisible to everything else. Its surviving chunk rows are the
    marker (§3.3), and the startup pass has to act on them.
    """
    first = await ingestion.ingest(request_of(document_of(5), source="notes.md"))
    async with database.session() as session:
        # Exactly the state a crash between the soft delete and the purge leaves.
        await DocumentRepository().soft_delete(session, first.document_id)

    stale_before = await vectors.query(
        registry.default.collection,
        _vector("content0 " * 10),
        top_k=50,
        where={"document_id": str(first.document_id)},
    )
    assert stale_before, "precondition: the vectors are still there"

    report = await ReconciliationService(
        sessions=database,
        documents=DocumentRepository(),
        chunks=ChunkRepository(),
        vectors=vectors,
        providers=registry,
    ).run()

    assert report.purged == 1
    stale_after = await vectors.query(
        registry.default.collection,
        _vector("content0 " * 10),
        top_k=50,
        where={"document_id": str(first.document_id)},
    )
    assert stale_after == ()
    async with database.session() as session:
        assert await ChunkRepository().count_for_document(session, first.document_id) == 0


async def test_a_clean_database_reconciles_to_nothing(
    database: Database, vectors: ChromaVectorStore, registry: ProviderRegistry
) -> None:
    report = await ReconciliationService(
        sessions=database,
        documents=DocumentRepository(),
        chunks=ChunkRepository(),
        vectors=vectors,
        providers=registry,
    ).run()

    assert report.clean
