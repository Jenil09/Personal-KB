"""`ChunkRepository` against a real Postgres.

The lookup that matters here is `find_reusable`. It decides how much of a
re-ingest costs money and minutes (AD-008), and every way it can be wrong —
crossing a collection boundary, matching a deleted document, missing a hash it
should have found — is a wrong answer that still looks like a working system.
"""

import pytest

from kb_api.adapters.postgres import ChunkRepository, DocumentRepository
from kb_api.domain import DocumentStatus
from platform_db import Database

pytestmark = pytest.mark.integration

SECOND_COLLECTION = "kb__gemini__gemini_embedding_001__1536__c1"


@pytest.fixture
def chunks_repo() -> ChunkRepository:
    return ChunkRepository()


@pytest.fixture
def documents_repo() -> DocumentRepository:
    return DocumentRepository()


async def test_a_batch_of_chunks_round_trips_in_order(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    document = make_document()

    async with database.session() as session:
        await documents_repo.add(session, document)
        written = await chunks_repo.add_all(
            session,
            [make_chunk(document.id, ordinal, f"paragraph {ordinal}") for ordinal in range(4)],
        )

    async with database.session() as session:
        stored = await chunks_repo.for_document(session, document.id)

    assert written == 4
    assert [chunk.ordinal for chunk in stored] == [0, 1, 2, 3]
    assert stored[2].text == "paragraph 2"


async def test_an_empty_batch_is_a_no_op(database: Database, chunks_repo: ChunkRepository) -> None:
    async with database.session() as session:
        assert await chunks_repo.add_all(session, []) == 0


async def test_documents_and_chunks_commit_together(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """§3.1 step 6 writes both in one transaction; a failure must leave neither.

    This is why the repositories take a session instead of holding one — a
    repository that opened its own scope could not participate in this.
    """
    document = make_document()

    with pytest.raises(RuntimeError):
        async with database.session() as session:
            await documents_repo.add(session, document)
            await chunks_repo.add_all(session, [make_chunk(document.id, 0, "orphan")])
            raise RuntimeError("chroma upsert failed")

    async with database.session() as session:
        assert await documents_repo.get(session, document.id) is None
        assert await chunks_repo.count_for_document(session, document.id) == 0


async def test_reusable_chunks_are_found_by_hash(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """The AD-008 carry-forward: an edited document re-embeds only what changed."""
    document = make_document()
    original = [make_chunk(document.id, ordinal, f"paragraph {ordinal}") for ordinal in range(3)]

    async with database.session() as session:
        await documents_repo.add(session, document)
        await chunks_repo.add_all(session, original)

    # Re-chunking after an edit to the middle paragraph: two hashes unchanged.
    edited = [
        original[0].text_hash,
        make_chunk(document.id, 1, "paragraph 1 rewritten").text_hash,
        original[2].text_hash,
    ]

    async with database.session() as session:
        reusable = await chunks_repo.find_reusable(session, edited, document.collection)

    assert set(reusable) == {original[0].text_hash, original[2].text_hash}
    assert reusable[original[0].text_hash].chroma_id == original[0].chroma_id


async def test_reuse_does_not_cross_a_collection_boundary(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """A `chroma_id` from another collection points at a vector that is not there.

    The hash covers the model but not which collection holds the vector, so the
    join to the parent document is the only thing preventing a reused chunk that
    search would never find.
    """
    document = make_document()
    chunk = make_chunk(document.id, 0, "shared paragraph")

    async with database.session() as session:
        await documents_repo.add(session, document)
        await chunks_repo.add_all(session, [chunk])

    async with database.session() as session:
        elsewhere = await chunks_repo.find_reusable(session, [chunk.text_hash], SECOND_COLLECTION)
        here = await chunks_repo.find_reusable(session, [chunk.text_hash], document.collection)

    assert elsewhere == {}
    assert set(here) == {chunk.text_hash}


async def test_chunks_of_a_deleted_document_are_not_reusable(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """Its vectors were purged from Chroma, so the hash matches nothing real."""
    document = make_document()
    chunk = make_chunk(document.id, 0, "paragraph")

    async with database.session() as session:
        await documents_repo.add(session, document)
        await chunks_repo.add_all(session, [chunk])
        await documents_repo.soft_delete(session, document.id)

    async with database.session() as session:
        reusable = await chunks_repo.find_reusable(session, [chunk.text_hash], document.collection)

    assert reusable == {}


async def test_an_empty_hash_list_asks_the_database_nothing(
    database: Database, chunks_repo: ChunkRepository
) -> None:
    """A document whose chunks are all new should not cost a query at all."""
    async with database.session() as session:
        assert await chunks_repo.find_reusable(session, [], "any-collection") == {}


async def test_deleting_a_document_takes_its_chunks(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """The `ON DELETE CASCADE`, which is what keeps a hard delete from orphaning rows."""
    document = make_document()

    async with database.session() as session:
        await documents_repo.add(session, document)
        await chunks_repo.add_all(session, [make_chunk(document.id, 0, "paragraph")])

    async with database.session() as session:
        await documents_repo.hard_delete(session, document.id)

    async with database.session() as session:
        assert await chunks_repo.count_for_document(session, document.id) == 0


async def test_chunks_can_be_deleted_without_the_document(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """The soft-delete flow: the document row stays as the crash-safety marker."""
    document = make_document()

    async with database.session() as session:
        await documents_repo.add(session, document)
        await chunks_repo.add_all(
            session, [make_chunk(document.id, ordinal, f"p{ordinal}") for ordinal in range(3)]
        )

    async with database.session() as session:
        removed = await chunks_repo.delete_for_document(session, document.id)

    async with database.session() as session:
        still_there = await documents_repo.get(session, document.id, include_deleted=True)

    assert removed == 3
    assert still_there is not None


async def test_two_chunks_cannot_share_an_ordinal(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """Ordinals are the chunk's position; a duplicate makes the order ambiguous."""
    from sqlalchemy.exc import IntegrityError

    document = make_document()

    async with database.session() as session:
        await documents_repo.add(session, document)
        await chunks_repo.add_all(session, [make_chunk(document.id, 0, "first")])

    with pytest.raises(IntegrityError):
        async with database.session() as session:
            await chunks_repo.add_all(session, [make_chunk(document.id, 0, "also first")])


async def test_status_and_chunk_count_describe_what_was_written(
    database: Database,
    chunks_repo: ChunkRepository,
    documents_repo: DocumentRepository,
    make_document,
    make_chunk,
) -> None:
    """`chunk_count` is denormalised, so it is worth checking it matches reality."""
    document = make_document()
    written = [make_chunk(document.id, ordinal, f"p{ordinal}") for ordinal in range(7)]

    async with database.session() as session:
        await documents_repo.add(session, document)
        count = await chunks_repo.add_all(session, written)
        await documents_repo.set_status(
            session, document.id, DocumentStatus.INDEXED, chunk_count=count
        )

    async with database.session() as session:
        stored = await documents_repo.get(session, document.id)
        actual = await chunks_repo.count_for_document(session, document.id)

    assert stored is not None
    assert stored.chunk_count == actual == 7
