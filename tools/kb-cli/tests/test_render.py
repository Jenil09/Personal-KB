"""What the printed output actually says.

Only the distinctions that carry meaning are asserted. Column widths and border
styles are not contract; the difference between "unchanged" and "ingested", and
between a collection holding zero vectors and one that could not be reached, are.
"""

import io
from datetime import UTC, datetime
from uuid import uuid4

from rich.console import Console

from kb_cli import render
from kb_cli.models import (
    CollectionStats,
    DocumentDetail,
    DocumentPage,
    DocumentSummary,
    IngestResult,
    SearchHit,
    SearchHitMetadata,
    SearchResponse,
    Stats,
    TokenUsage,
)
from platform_core import UpstreamError

STAMP = datetime(2026, 7, 30, 9, 12, tzinfo=UTC)


def text_of(renderable: object) -> str:
    """Render to a string at a width wide enough not to wrap the assertions."""
    console = Console(width=220, file=io.StringIO(), record=True)
    console.print(renderable)
    return console.export_text()


def summary(**overrides: object) -> DocumentSummary:
    values: dict[str, object] = {
        "id": uuid4(),
        "title": "Redshift Architecture",
        "type": "architecture",
        "collection": "kb__openai__c1",
        "status": "indexed",
        "chunk_count": 14,
        "content_hash": "9f2c",
        "created_at": STAMP,
        "updated_at": STAMP,
        "source": "redshift.md",
        "tags": ("ansible", "hardening"),
    }
    values.update(overrides)
    return DocumentSummary.model_validate(values)


def page(*documents: DocumentSummary, total: int | None = None, offset: int = 0) -> DocumentPage:
    return DocumentPage(
        documents=documents,
        total=len(documents) if total is None else total,
        limit=50,
        offset=offset,
    )


# --- listings -------------------------------------------------------------


def test_the_table_names_each_document() -> None:
    rendered = text_of(render.documents_table(page(summary(), summary(title="Runbook"))))

    assert "Redshift Architecture" in rendered
    assert "Runbook" in rendered


def test_an_empty_listing_says_so_rather_than_printing_an_empty_table() -> None:
    assert "No documents match." in text_of(render.documents_table(page()))


def test_the_caption_reports_the_position_in_the_whole_result_set() -> None:
    rendered = text_of(render.documents_table(page(summary(), total=24)))

    assert "of 24" in rendered


def test_a_partial_listing_says_how_to_get_the_rest() -> None:
    """`total` is the count matching the filters, not the page size."""
    rendered = text_of(render.documents_table(page(summary(), total=24)))

    assert "--offset" in rendered


def test_a_complete_listing_does_not_suggest_paging() -> None:
    assert "--offset" not in text_of(render.documents_table(page(summary())))


def test_ids_are_shortened_consistently() -> None:
    document = summary()

    assert render.short_id(document.id) == str(document.id)[:8]
    assert render.short_id(document.id) in text_of(render.documents_table(page(document)))


# --- detail ---------------------------------------------------------------


def detail(**overrides: object) -> DocumentDetail:
    values = summary().model_dump()
    values.update({"content": "# Redshift\n\nthe body text"})
    values.update(overrides)
    return DocumentDetail.model_validate(values)


def test_the_detail_panel_shows_the_content() -> None:
    assert "the body text" in text_of(render.document_panel(detail()))


def test_the_detail_panel_can_omit_the_content() -> None:
    rendered = text_of(render.document_panel(detail(), content=False))

    assert "the body text" not in rendered
    assert "indexed" in rendered


def test_provenance_appears_only_when_it_exists() -> None:
    """A document ingested before the columns existed has neither (AD-014)."""
    with_provenance = text_of(render.document_panel(detail(ingested_by_key_id="cli")))
    without = text_of(render.document_panel(detail()))

    assert "ingested by" in with_provenance
    assert "ingested by" not in without


# --- ingest results -------------------------------------------------------


def ingest_result(**overrides: object) -> IngestResult:
    values: dict[str, object] = {
        "document_id": uuid4(),
        "chunks_created": 14,
        "chunks_reused": 0,
        "total_tokens": 3842,
        "status": "success",
        "collection": "kb__openai__c1",
        "superseded": (),
    }
    values.update(overrides)
    return IngestResult.model_validate(values)


def test_a_fresh_ingest_reports_what_it_embedded() -> None:
    rendered = text_of(render.ingest_result_line("notes.md", ingest_result()))

    assert "ingested" in rendered
    assert "3,842 tokens" in rendered


def test_an_unchanged_ingest_is_visibly_different() -> None:
    """AD-008: a re-run that cost nothing must not look like one that cost an embed."""
    rendered = text_of(
        render.ingest_result_line(
            "notes.md", ingest_result(status="unchanged", chunks_created=0, chunks_reused=14)
        )
    )

    assert "unchanged" in rendered
    assert "3,842 tokens" not in rendered


def test_a_supersede_is_reported() -> None:
    """AD-020: a replacement must not read the same as a first ingest."""
    rendered = text_of(render.ingest_result_line("notes.md", ingest_result(superseded=(uuid4(),))))

    assert "replaced 1" in rendered


# --- search ---------------------------------------------------------------


def hit(score: float = 0.87, title: str = "Redshift Architecture") -> SearchHit:
    return SearchHit(
        id="doc:7",
        text="Bare metal Kubernetes with Cilium and Longhorn. " * 20,
        metadata=SearchHitMetadata(
            document_id=str(uuid4()), title=title, type="architecture", ordinal=7
        ),
        score=score,
    )


def test_search_results_are_ranked_and_scored() -> None:
    response = SearchResponse(
        results=(hit(0.9), hit(0.4, "Other")), query_tokens=18, latency_ms=412
    )

    rendered = text_of(render.search_results(response))

    assert "1." in rendered
    assert "0.900" in rendered
    assert "412 ms" in rendered


def test_search_excerpts_are_truncated_by_default() -> None:
    rendered = text_of(render.search_results(SearchResponse(results=(hit(),))))

    assert "…" in rendered


def test_full_text_prints_the_whole_chunk() -> None:
    matched = hit()

    rendered = text_of(render.search_results(SearchResponse(results=(matched,)), full_text=True))

    assert rendered.count("Bare metal Kubernetes") > 1


def test_no_results_says_so() -> None:
    assert "No results." in text_of(render.search_results(SearchResponse(results=())))


# --- stats ----------------------------------------------------------------


def stats(**overrides: object) -> Stats:
    values: dict[str, object] = {
        "documents_by_status": {"indexed": 24},
        "total_documents": 24,
        "total_chunks": 6700,
        "total_tokens_stored": 2_980_000,
        "collections": (
            CollectionStats(
                name="kb__openai__c1",
                provider="openai",
                model="text-embedding-3-small",
                dimensions=1536,
                vectors=6700,
            ),
        ),
        "tokens": TokenUsage(exact_tokens=2_980_000, api_calls=68),
    }
    values.update(overrides)
    return Stats.model_validate(values)


def test_the_status_view_reports_the_corpus() -> None:
    rendered = text_of(render.stats_view(stats()))

    assert "6,700" in rendered
    assert "text-embedding-3-small" in rendered


def test_an_unreachable_collection_is_distinct_from_an_empty_one() -> None:
    """`null` vectors mean Chroma could not be reached; `0` means empty."""
    unreachable = text_of(
        render.stats_view(
            stats(
                collections=(
                    CollectionStats(
                        name="c", provider="openai", model="m", dimensions=1536, vectors=None
                    ),
                )
            )
        )
    )

    assert "unreachable" in unreachable


def test_a_spilled_audit_queue_is_surfaced() -> None:
    """Non-zero here is what `/health` reports as `degraded` (AD-013)."""
    assert stats(audit_spill_depth=3).degraded is True
    assert "3" in text_of(render.stats_view(stats(audit_spill_depth=3)))


def test_a_healthy_service_is_not_degraded() -> None:
    assert stats().degraded is False


# --- errors ---------------------------------------------------------------


def test_an_error_prints_the_services_own_sentence() -> None:
    rendered = text_of(render.error_text(UpstreamError("chroma is unreachable")))

    assert "chroma is unreachable" in rendered


def test_an_error_prints_its_context_because_that_is_the_actionable_part() -> None:
    """The required scope on a 403 is what says what to do about it."""
    rendered = text_of(
        render.error_text(UpstreamError("nope", context={"required_scope": "write"}))
    )

    assert "required_scope" in rendered
    assert "write" in rendered


def test_an_error_prints_the_request_id_for_the_audit_trail() -> None:
    rendered = text_of(render.error_text(UpstreamError("nope", context={"request_id": "abc-123"})))

    assert "abc-123" in rendered


def test_a_plain_exception_still_renders() -> None:
    """`main` hands this anything; a missing `context` must not raise here."""
    assert "boom" in text_of(render.error_text(ValueError("boom")))


# --- shared helpers -------------------------------------------------------


def test_bytes_are_humanised() -> None:
    assert render.humanise_bytes(512) == "512 B"
    assert render.humanise_bytes(2048) == "2.0 KB"
    assert render.humanise_bytes(5 * 1024 * 1024) == "5.0 MB"
