"""The Chroma adapter against a real Chroma.

A stubbed Chroma would test the stub. Everything asserted here is a property of
the server rather than of the code calling it: that the collection is created in
cosine space, that a metadata filter deletes the right vectors and only those,
that an upsert of an existing id overwrites instead of duplicating, and that a
missing collection is a specific failure rather than an exception with the
adapter's name on it.
"""

import math
from uuid import UUID, uuid4

import pytest
from testcontainers.community.chroma import ChromaContainer

from kb_api.adapters.chroma import ChromaVectorStore, create_chroma_client
from kb_api.domain import VectorRecord, chunk_metadata
from platform_core import ConflictError

pytestmark = pytest.mark.integration

DIMENSIONS = 8


def basis(index: int) -> tuple[float, ...]:
    """The `index`-th unit vector. Self-identifying, and already normalised."""
    return tuple(1.0 if position == index else 0.0 for position in range(DIMENSIONS))


def record(document_id: UUID, ordinal: int, *, title: str = "Doc") -> VectorRecord:
    return VectorRecord(
        id=f"{document_id}:{ordinal}",
        vector=basis(ordinal),
        chunk_text=f"chunk {ordinal} of {title}",
        metadata=chunk_metadata(
            document_id=document_id,
            title=title,
            source=f"{title.lower()}.md",
            document_type="architecture",
            tags=("ansible", "hardening"),
            ordinal=ordinal,
        ),
    )


@pytest.fixture
async def store(chroma: ChromaContainer, chroma_port: int) -> ChromaVectorStore:
    instance = ChromaVectorStore(
        lambda: create_chroma_client(
            host="127.0.0.1",
            port=chroma_port,
            tenant="default_tenant",
            database="default_database",
        )
    )
    await instance.connect()
    return instance


@pytest.fixture
def collection() -> str:
    """A fresh name per test. Chroma has no truncate, and a shared collection
    would make every test depend on the order the others ran in."""
    return f"kb__test__basis__{DIMENSIONS}__c1_{uuid4().hex[:8]}"


async def test_creating_a_collection_is_idempotent(
    store: ChromaVectorStore, collection: str
) -> None:
    await store.ensure_collection(collection)
    await store.ensure_collection(collection)

    assert await store.collection_exists(collection)


async def test_a_collection_that_was_never_created_does_not_exist(
    store: ChromaVectorStore, collection: str
) -> None:
    assert not await store.collection_exists(collection)


async def test_the_collection_is_cosine_space(store: ChromaVectorStore, collection: str) -> None:
    # Design §2.3, and the reason search can report `1 - distance` as a
    # similarity. Under the default L2 space that arithmetic yields a number
    # that is not a similarity at all.
    await store.ensure_collection(collection)
    document_id = uuid4()
    await store.upsert(collection, [record(document_id, 0), record(document_id, 1)])

    matches = await store.query(collection, basis(0), top_k=2)

    # Cosine distance: 0 for the identical vector, 1 for an orthogonal one.
    assert math.isclose(matches[0].distance, 0.0, abs_tol=1e-6)
    assert math.isclose(matches[1].distance, 1.0, abs_tol=1e-6)


async def test_a_query_returns_the_text_and_metadata_the_response_is_built_from(
    store: ChromaVectorStore, collection: str
) -> None:
    # AD-004: the search response comes from the Chroma payload alone, so
    # anything missing here is a field the API cannot report at all.
    await store.ensure_collection(collection)
    document_id = uuid4()
    await store.upsert(collection, [record(document_id, 0, title="Redshift")])

    (match,) = await store.query(collection, basis(0), top_k=1)

    assert match.chunk_text == "chunk 0 of Redshift"
    assert match.metadata["document_id"] == str(document_id)
    assert match.metadata["tags"] == "ansible|hardening"
    assert match.metadata["ordinal"] == 0


async def test_upserting_the_same_id_overwrites_rather_than_duplicates(
    store: ChromaVectorStore, collection: str
) -> None:
    # This is what makes replaying an interrupted ingest safe (§3.1). If it
    # appended, every crash-recovery pass would double the collection.
    await store.ensure_collection(collection)
    document_id = uuid4()
    await store.upsert(collection, [record(document_id, 0)])
    await store.upsert(collection, [record(document_id, 0, title="Rewritten")])

    assert await store.count(collection) == 1
    (match,) = await store.query(collection, basis(0), top_k=5)
    assert match.chunk_text == "chunk 0 of Rewritten"


async def test_deleting_a_document_removes_its_vectors_and_no_others(
    store: ChromaVectorStore, collection: str
) -> None:
    await store.ensure_collection(collection)
    doomed, survivor = uuid4(), uuid4()
    await store.upsert(collection, [record(doomed, 0), record(doomed, 1)])
    await store.upsert(collection, [record(survivor, 2)])

    deleted = await store.delete_document(collection, doomed)

    assert deleted == 2
    assert await store.count(collection) == 1
    (match,) = await store.query(collection, basis(2), top_k=5)
    assert match.metadata["document_id"] == str(survivor)


async def test_deleting_from_a_collection_that_does_not_exist_is_a_success(
    store: ChromaVectorStore, collection: str
) -> None:
    # Design §3.3 makes delete idempotent, and "the collection was never
    # created" is a stronger form of "there is nothing to delete".
    assert await store.delete_document(collection, uuid4()) == 0


async def test_fetching_vectors_returns_what_was_stored(
    store: ChromaVectorStore, collection: str
) -> None:
    # The AD-019 carry-forward path. A vector that comes back changed would
    # silently move every reused chunk in the index.
    await store.ensure_collection(collection)
    document_id = uuid4()
    stored = record(document_id, 3)
    await store.upsert(collection, [stored])

    found = await store.fetch_vectors(collection, [stored.id])

    assert set(found) == {stored.id}
    assert all(
        math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(found[stored.id], stored.vector, strict=True)
    )


async def test_fetching_an_absent_id_omits_it_rather_than_failing(
    store: ChromaVectorStore, collection: str
) -> None:
    # Drift between the stores is recoverable — AD-003 says Postgres wins and
    # the index is rebuildable — so the caller re-embeds what is missing.
    await store.ensure_collection(collection)
    document_id = uuid4()
    stored = record(document_id, 0)
    await store.upsert(collection, [stored])

    found = await store.fetch_vectors(collection, [stored.id, "not-a-stored-id"])

    assert set(found) == {stored.id}


async def test_an_empty_upsert_costs_nothing(store: ChromaVectorStore, collection: str) -> None:
    # A re-ingest where every chunk was carried forward has no new vectors, and
    # Chroma answers an empty upsert with a 400.
    await store.ensure_collection(collection)

    await store.upsert(collection, [])

    assert await store.count(collection) == 0


async def test_counting_a_collection_that_does_not_exist_is_zero(
    store: ChromaVectorStore, collection: str
) -> None:
    assert await store.count(collection) == 0


async def test_querying_a_collection_that_does_not_exist_is_a_conflict(
    store: ChromaVectorStore, collection: str
) -> None:
    # A `409`, which is what Design §3.2 answers for a provider naming a
    # collection with nothing in it — not an unhandled `500`, and not
    # `chromadb.errors.NotFoundError` escaping into the service.
    with pytest.raises(ConflictError) as raised:
        await store.query(collection, basis(0), top_k=1)

    assert raised.value.status_code == 409


async def test_a_scalar_where_clause_filters(store: ChromaVectorStore, collection: str) -> None:
    # AD-005: `type` and `source` are pushed into Chroma directly; only tags
    # have to detour through Postgres.
    await store.ensure_collection(collection)
    wanted, other = uuid4(), uuid4()
    await store.upsert(collection, [record(wanted, 0, title="Wanted")])
    await store.upsert(collection, [record(other, 1, title="Other")])

    matches = await store.query(collection, basis(0), top_k=5, where={"source": "wanted.md"})

    assert [match.metadata["document_id"] for match in matches] == [str(wanted)]


async def test_the_heartbeat_answers(store: ChromaVectorStore) -> None:
    assert await store.heartbeat()
