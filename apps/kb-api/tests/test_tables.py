"""Schema shape, without a database.

These are the claims that need to hold before a container is worth starting: the
metadata is in the right schemas, and the tier-1 table comes from the library
rather than being redeclared here (AD-018). A second definition of
`request_logs` would migrate cleanly and diverge from the writer silently, which
is the failure AD-018 exists to prevent and the one integration tests would not
catch — they would be exercising the redefinition.
"""

from kb_api.adapters.postgres import (
    KB_SCHEMA,
    chunks,
    documents,
    error_logs,
    ingest_logs,
    kb_metadata,
    telemetry_metadata,
    token_usage_logs,
)
from platform_db import AUDIT_SCHEMA, audit_metadata, request_logs


def test_the_knowledge_base_tables_live_in_the_kb_schema() -> None:
    assert {table.name for table in kb_metadata.sorted_tables} == {"documents", "chunks"}
    assert documents.schema == KB_SCHEMA
    assert chunks.schema == KB_SCHEMA


def test_the_telemetry_tables_live_in_the_audit_schema() -> None:
    assert {table.name for table in telemetry_metadata.sorted_tables} == {
        "token_usage_logs",
        "ingest_logs",
        "error_logs",
    }
    for table in (token_usage_logs, ingest_logs, error_logs):
        assert table.schema == AUDIT_SCHEMA


def test_the_service_does_not_redefine_the_tier_one_table() -> None:
    """AD-018: one owner for the writer, the spill format, and the columns."""
    ours = {table.name for table in (*kb_metadata.sorted_tables, *telemetry_metadata.sorted_tables)}

    assert "request_logs" not in ours
    assert request_logs.metadata is audit_metadata


def test_documents_and_chunks_are_not_in_the_same_metadata_as_the_library() -> None:
    """Separate objects, so a service cannot mutate the library's migration story."""
    assert kb_metadata is not audit_metadata
    assert telemetry_metadata is not audit_metadata


def test_the_content_hash_pair_is_unique_only_among_live_documents() -> None:
    """Deleting and resubmitting the same content has to stay possible."""
    index = next(i for i in documents.indexes if i.name == "uq_documents_content_hash_collection")

    assert index.unique
    assert index.dialect_options["postgresql"]["where"] is not None


def test_the_tag_index_is_gin_with_the_containment_opclass() -> None:
    """AD-005's lookup is `@>` and nothing else, which is what jsonb_path_ops covers."""
    index = next(i for i in documents.indexes if i.name == "ix_documents_tags")
    options = index.dialect_options["postgresql"]

    assert options["using"] == "gin"
    assert options["ops"] == {"tags": "jsonb_path_ops"}
