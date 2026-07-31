"""AD-005's tag lookup: correct results, and reached through the GIN index.

Two claims, and the second is the one that rots quietly. A tag query that has
stopped using `ix_documents_tags` still returns the right documents — it just
reads the whole table to do it, and at a few dozen documents nobody notices until
the corpus is not a few dozen documents.

So the table is seeded past the point where the planner has a genuine choice.
Postgres will sequentially scan a small table however good the index is, and
correctly: an `EXPLAIN` assertion against ten rows would either fail for the
wrong reason or be forced to pass with `enable_seqscan = off`, which proves only
that the index *can* be used, not that it *is*.
"""

import json

import pytest
from sqlalchemy import Select, select, text

from kb_api.adapters.postgres import DocumentRepository
from kb_api.adapters.postgres.documents import _ALIVE, _conditions
from kb_api.adapters.postgres.tables import documents
from kb_api.domain import DocumentFilter
from platform_db import Database
from platform_db.testing import plan_for, uses_index

pytestmark = pytest.mark.integration

# Enough that a sequential scan is the more expensive plan. The real corpus
# ceiling is ~100 documents (Design §8), so this is deliberately far past it:
# the point is to make the planner's choice meaningful, not to model the corpus.
SEEDED = 4_000

VOCABULARY = ("ansible", "hardening", "provisioning", "networking", "storage", "backup")

# Rare on purpose. A tag matching half the table is one the planner is right to
# answer with a sequential scan, so a selective predicate is what makes the
# assertion about the index rather than about the seed data.
RARE_TAG = "cilium"

TYPES = ("architecture", "runbook", "reference")


@pytest.fixture
def repository() -> DocumentRepository:
    return DocumentRepository()


@pytest.fixture
async def seeded(database: Database, make_document) -> dict[str, tuple[str, ...]]:
    """`SEEDED` documents with deterministic tags. Returns the expected mapping.

    Built with one executemany and a hand-made row list rather than
    `repository.add` in a loop: 4,000 round trips would dominate the suite's
    runtime, and what is under test here is the read path.
    """
    expected: dict[str, tuple[str, ...]] = {}
    rows = []
    for index in range(SEEDED):
        # Deterministic, not random: a flaky index test that only reproduces
        # under one seed is worse than no index test.
        tags = tuple(VOCABULARY[(index + offset) % len(VOCABULARY)] for offset in range(2))
        if index % 500 == 0:
            tags = (*tags, RARE_TAG)
        document = make_document(name=f"seed-{index}", tags=tags)
        expected[str(document.id)] = tags
        rows.append(
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "content_hash": document.content_hash,
                "type": TYPES[index % len(TYPES)],
                "tags": list(tags),
                "collection": document.collection,
                "status": "indexed",
                "chunk_count": 1,
            }
        )

    async with database.engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO kb.documents "
                "(id, title, content, content_hash, type, tags, collection, status, chunk_count) "
                "VALUES (:id, :title, :content, :content_hash, :type, "
                "CAST(:tags AS jsonb), :collection, :status, :chunk_count)"
            ),
            [{**row, "tags": json.dumps(row["tags"])} for row in rows],
        )
        # Without statistics the planner is working from defaults and its
        # choice says nothing about the data that is actually there.
        await connection.execute(text("ANALYZE kb.documents"))

    return expected


async def test_a_tag_query_matches_a_naive_in_memory_filter(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
) -> None:
    """Design §7's non-negotiable case: the same set, computed two ways."""
    wanted = "networking"

    async with database.session() as session:
        found = await repository.ids_matching(session, DocumentFilter(tags=(wanted,)))

    expected = {doc_id for doc_id, tags in seeded.items() if wanted in tags}

    assert {str(doc_id) for doc_id in found} == expected
    assert expected


async def test_several_tags_match_any_of_them_by_default(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
) -> None:
    """Any-of is the semantics AD-005 was defending when it rejected boolean keys."""
    wanted = ("storage", RARE_TAG)

    async with database.session() as session:
        found = await repository.ids_matching(session, DocumentFilter(tags=wanted))

    expected = {doc_id for doc_id, tags in seeded.items() if set(wanted) & set(tags)}

    assert {str(doc_id) for doc_id in found} == expected


async def test_match_all_tags_returns_the_intersection(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
) -> None:
    wanted = ("storage", RARE_TAG)

    async with database.session() as session:
        found = await repository.ids_matching(
            session, DocumentFilter(tags=wanted, match_all_tags=True)
        )

    expected = {doc_id for doc_id, tags in seeded.items() if set(wanted) <= set(tags)}

    assert {str(doc_id) for doc_id in found} == expected
    # A meaningful intersection: strictly smaller than either tag alone, and
    # not empty, or the assertion above would hold trivially.
    assert 0 < len(expected) < sum(1 for tags in seeded.values() if "storage" in tags)


async def test_the_tag_query_uses_the_gin_index(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
) -> None:
    """The exit criterion: verified by `EXPLAIN`, on a table big enough to choose.

    Explained through the repository's own statement, binds and all — a
    hand-written `EXPLAIN` string would prove an index covers a query that
    nothing in the service actually runs.
    """
    async with database.session() as session:
        statement = _tag_statement(DocumentFilter(tags=(RARE_TAG,)))
        plan = await plan_for(session, statement)

    assert uses_index(plan, "ix_documents_tags"), plan
    assert "Seq Scan on documents" not in plan, plan


async def test_the_multi_tag_predicate_can_be_answered_by_the_index(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
) -> None:
    """Any-of is an OR of containments, and each side has to stay indexable.

    Asserted with sequential scans discouraged rather than as a free planner
    choice, because at this size the free choice is legitimately a sequential
    scan: two tags per row make the GIN index big enough that a `BitmapOr` of
    two scans costs more than reading 4,000 short rows, and Postgres is right
    about that. Pinning the plan anyway would be asserting a cost accident.

    What survives table size is whether the predicate is *answerable* by the
    index at all — the thing a stray cast or a function wrapped around `tags`
    would silently destroy. With `enable_seqscan` off Postgres still falls back
    to a sequential scan when no index can serve the predicate, so this fails
    exactly when it should.
    """
    async with database.session() as session:
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        statement = _tag_statement(DocumentFilter(tags=(RARE_TAG, "nonexistent-tag")))
        plan = await plan_for(session, statement)

    assert uses_index(plan, "ix_documents_tags"), plan
    assert "BitmapOr" in plan, plan


async def test_a_deleted_document_drops_out_of_tag_results(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
    make_document,
) -> None:
    """Otherwise a filtered search hands Chroma an ID whose vectors are gone."""
    async with database.session() as session:
        before = await repository.ids_matching(session, DocumentFilter(tags=(RARE_TAG,)))

    async with database.session() as session:
        await repository.soft_delete(session, before[0])

    async with database.session() as session:
        after = await repository.ids_matching(session, DocumentFilter(tags=(RARE_TAG,)))

    assert set(after) == set(before) - {before[0]}


async def test_the_id_list_can_be_capped(
    database: Database,
    repository: DocumentRepository,
    seeded: dict[str, tuple[str, ...]],
) -> None:
    """AD-005 notes the `$in` clause gets unwieldy past a few thousand IDs."""
    async with database.session() as session:
        found = await repository.ids_matching(
            session, DocumentFilter(tags=("networking",)), limit=10
        )

    assert len(found) == 10


def _tag_statement(filters: DocumentFilter) -> Select[tuple[object]]:
    """The statement `ids_matching` runs, without running it.

    Reaching for the private builder is deliberate: the alternative is
    re-deriving the `where` clause in the test, and then the plan under
    assertion is not the plan the repository produces.
    """
    return select(documents.c.id).where(_ALIVE, *_conditions(filters))
