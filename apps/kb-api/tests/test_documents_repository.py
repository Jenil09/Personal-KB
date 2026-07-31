"""`DocumentRepository` against a real Postgres.

Behaviour, not statements: what these assert is that a re-ingest finds the
existing document, that a deleted one stops being found, and that the
constraints in the migration actually hold — the sort of thing a mocked session
would agree with regardless of whether the schema says so.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from kb_api.adapters.postgres import DocumentRepository
from kb_api.adapters.postgres import documents as documents_table
from kb_api.domain import DocumentFilter, DocumentStatus
from platform_db import Database

pytestmark = pytest.mark.integration

SECOND_COLLECTION = "kb__gemini__gemini_embedding_001__1536__c1"


@pytest.fixture
def repository() -> DocumentRepository:
    return DocumentRepository()


async def test_a_document_round_trips(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    document = make_document(tags=("ansible", "hardening"), source="arch.md")

    async with database.session() as session:
        stored = await repository.add(session, document)

    async with database.session() as session:
        found = await repository.get(session, document.id)

    assert stored.id == document.id
    assert stored.status is DocumentStatus.PENDING
    assert stored.created_at is not None
    assert found == stored
    assert found is not None
    assert found.tags == ("ansible", "hardening")


async def test_re_ingesting_identical_content_finds_the_existing_document(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """AD-008's short-circuit, which is what makes a repeat ingest cost nothing."""
    document = make_document()

    async with database.session() as session:
        await repository.add(session, document)

    async with database.session() as session:
        found = await repository.find_by_content_hash(
            session, document.content_hash, document.collection
        )

    assert found is not None
    assert found.id == document.id


async def test_the_same_content_in_another_collection_is_not_a_duplicate(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """A second provider's collection has no vectors for it, so it is un-ingested (AD-006).

    Also the case the partial unique index has to permit: same `content_hash`,
    different `collection`, both live.
    """
    first = make_document()
    second = make_document(collection=SECOND_COLLECTION)

    async with database.session() as session:
        await repository.add(session, first)
        await repository.add(session, second)

    async with database.session() as session:
        found = await repository.find_by_content_hash(
            session, first.content_hash, SECOND_COLLECTION
        )

    assert first.content_hash == second.content_hash
    assert found is not None
    assert found.id == second.id


async def test_duplicate_content_in_one_collection_is_rejected(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """The unique index is the backstop for a race the hash check cannot cover.

    Two concurrent ingests of the same content both miss on the lookup; the
    index is what stops them both writing.
    """
    document = make_document()

    async with database.session() as session:
        await repository.add(session, document)

    with pytest.raises(IntegrityError):
        async with database.session() as session:
            await repository.add(
                session, make_document(name="different-name", content=document.content)
            )


async def test_content_resubmitted_after_a_delete_is_allowed(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """Why the unique index is partial on `deleted_at IS NULL`.

    Deleting a document and ingesting the same content again is an ordinary
    thing to do, and a total unique index would reject it forever.
    """
    original = make_document()

    async with database.session() as session:
        await repository.add(session, original)
        await repository.soft_delete(session, original.id)

    async with database.session() as session:
        resubmitted = await repository.add(
            session, make_document(name="resubmitted", content=original.content)
        )

    assert resubmitted.content_hash == original.content_hash
    assert resubmitted.id != original.id


async def test_a_soft_deleted_document_stops_being_found(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    document = make_document()

    async with database.session() as session:
        await repository.add(session, document)
        deleted = await repository.soft_delete(session, document.id)

    async with database.session() as session:
        visible = await repository.get(session, document.id)
        including_deleted = await repository.get(session, document.id, include_deleted=True)
        by_hash = await repository.find_by_content_hash(
            session, document.content_hash, document.collection
        )

    assert deleted is True
    assert visible is None
    assert by_hash is None
    # The row is still there, and still says who ingested it. A trail that
    # forgets deleted documents cannot answer the question AD-014 poses.
    assert including_deleted is not None
    assert including_deleted.is_deleted


async def test_deleting_twice_reports_the_second_call_did_nothing(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """What lets `DELETE` stay idempotent without a second query (Design §3.3)."""
    document = make_document()

    async with database.session() as session:
        await repository.add(session, document)

    async with database.session() as session:
        first = await repository.soft_delete(session, document.id)
        second = await repository.soft_delete(session, document.id)

    assert first is True
    assert second is False


async def test_marking_indexed_records_the_chunk_count(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    document = make_document()

    async with database.session() as session:
        stored = await repository.add(session, document)

    async with database.session() as session:
        updated = await repository.set_status(
            session, document.id, DocumentStatus.INDEXED, chunk_count=14
        )

    assert updated is not None
    assert updated.status is DocumentStatus.INDEXED
    assert updated.chunk_count == 14
    assert updated.updated_at >= stored.updated_at


async def test_pending_documents_are_what_reconciliation_sees(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """A crash between the Postgres commit and the Chroma upsert leaves these."""
    stuck = make_document(name="stuck")
    finished = make_document(name="finished")

    async with database.session() as session:
        await repository.add(session, stuck)
        await repository.add(session, finished)
        await repository.set_status(session, finished.id, DocumentStatus.INDEXED)

    async with database.session() as session:
        pending = await repository.find_pending(session)

    assert [document.id for document in pending] == [stuck.id]


async def test_provenance_survives_the_round_trip(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """AD-014: what this service offers against a malicious payload is traceability."""
    document = make_document(ingested_by_key_id="n8n", ingested_from_ip="203.0.113.7")

    async with database.session() as session:
        stored = await repository.add(session, document)

    assert stored.ingested_by_key_id == "n8n"
    assert str(stored.ingested_from_ip) == "203.0.113.7"


async def test_an_unknown_status_is_rejected_by_the_database(
    database: Database, make_document
) -> None:
    """The check constraint, not the enum, is what makes the column trustworthy.

    A bad status can only arrive through something bypassing `DocumentStatus`,
    which is exactly when a database-level guard earns its place.
    """
    document = make_document()

    with pytest.raises(IntegrityError):
        async with database.session() as session:
            await session.execute(
                documents_table.insert().values(
                    id=document.id,
                    title=document.title,
                    content=document.content,
                    content_hash=document.content_hash,
                    type=document.type,
                    collection=document.collection,
                    status="half-done",
                )
            )


async def test_listing_is_newest_first_and_paginates(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    async with database.session() as session:
        for index in range(5):
            await repository.add(session, make_document(name=f"document-{index}"))

    async with database.session() as session:
        first_page = await repository.list(session, limit=2)
        second_page = await repository.list(session, limit=2, offset=2)

    assert first_page.total == 5
    assert len(first_page.documents) == 2
    assert len(second_page.documents) == 2
    # No overlap: the ordering is total, so a page boundary cannot repeat a row.
    assert not {d.id for d in first_page.documents} & {d.id for d in second_page.documents}


async def test_listing_filters_by_type(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    async with database.session() as session:
        await repository.add(session, make_document(name="arch", type="architecture"))
        await repository.add(session, make_document(name="runbook", type="runbook"))

    async with database.session() as session:
        page = await repository.list(session, DocumentFilter(type="runbook"))

    assert page.total == 1
    assert page.documents[0].type == "runbook"


async def test_metadata_edits_do_not_touch_content(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    document = make_document(tags=("old",))

    async with database.session() as session:
        await repository.add(session, document)

    async with database.session() as session:
        updated = await repository.update_metadata(
            session, document.id, title="Renamed", tags=("new", "tags")
        )

    assert updated is not None
    assert updated.title == "Renamed"
    assert updated.tags == ("new", "tags")
    # Unchanged, so the vectors in Chroma still describe what Postgres holds.
    assert updated.content == document.content
    assert updated.content_hash == document.content_hash


async def test_updating_a_missing_document_reports_nothing_rather_than_raising(
    database: Database, repository: DocumentRepository, make_document
) -> None:
    """A missing row is a `None` here; whether that is a 404 is the service's call."""
    document = make_document()

    async with database.session() as session:
        updated = await repository.set_status(session, document.id, DocumentStatus.INDEXED)

    assert updated is None
