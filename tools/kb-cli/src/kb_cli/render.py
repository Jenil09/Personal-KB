"""Rich renderables for the subcommands.

Kept apart from `cli.py` so the command functions stay "call the client, hand
the result to a renderer" and the formatting can be exercised without invoking
Typer. The TUI does not use this module — Textual has its own widgets — but both
agree on the helpers at the bottom (`short_id`, `humanise_bytes`) so an id
truncated in `kb list` matches the one shown in the browser.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kb_cli.models import DocumentDetail, DocumentPage, IngestResult, SearchResponse, Stats

__all__ = [
    "console",
    "document_panel",
    "documents_table",
    "error_text",
    "ingest_result_line",
    "search_results",
    "short_id",
    "stats_view",
]

console = Console()
err_console = Console(stderr=True)

_ID_CHARS = 8


def short_id(value: UUID | str) -> str:
    """The first segment of a UUID.

    Enough to recognise a document in a list of tens and to paste back into
    `kb show`, which resolves a prefix. The full id is on the detail view, where
    there is room for it and where someone is about to copy it somewhere exact.
    """
    return str(value)[:_ID_CHARS]


def humanise_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def documents_table(page: DocumentPage, *, title: str | None = None) -> RenderableType:
    if not page.documents:
        return Text("No documents match.", style="dim")
    table = Table(
        title=title,
        title_justify="left",
        header_style="bold",
        expand=True,
        row_styles=("", "on grey11"),
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", ratio=3, overflow="ellipsis")
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Tags", ratio=2, overflow="ellipsis", style="dim")
    table.add_column("Chunks", justify="right", no_wrap=True)
    table.add_column("Updated", no_wrap=True, style="dim")
    for document in page.documents:
        table.add_row(
            short_id(document.id),
            document.title,
            document.type,
            ", ".join(document.tags),
            str(document.chunk_count),
            _stamp(document.updated_at),
        )
    shown = page.offset + len(page.documents)
    caption = f"{page.offset + 1}\u2013{shown} of {page.total}"
    if page.has_more:
        caption += f"  ·  more with --offset {shown}"
    table.caption = caption
    table.caption_justify = "right"
    return table


def document_panel(document: DocumentDetail, *, content: bool = True) -> RenderableType:
    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim", no_wrap=True)
    header.add_column()
    header.add_row("id", str(document.id))
    header.add_row("type", document.type)
    header.add_row("source", document.source or "—")
    header.add_row("tags", ", ".join(document.tags) or "—")
    header.add_row("collection", document.collection)
    header.add_row("status", document.status)
    header.add_row("chunks", str(document.chunk_count))
    header.add_row("size", humanise_bytes(len(document.content.encode())))
    header.add_row("created", _stamp(document.created_at))
    header.add_row("updated", _stamp(document.updated_at))
    # Provenance is on the detail response only (AD-014) and is printed only
    # when it exists — a document ingested before the columns did has neither,
    # and two empty rows would read as a service that stopped recording them.
    if document.ingested_by_key_id:
        header.add_row("ingested by", document.ingested_by_key_id)
    if document.ingested_from_ip:
        header.add_row("ingested from", document.ingested_from_ip)

    body: list[RenderableType] = [header]
    if content:
        body += [Text(""), Text(document.content)]
    return Panel(Group(*body), title=document.title, title_align="left", border_style="cyan")


def search_results(response: SearchResponse, *, full_text: bool = False) -> RenderableType:
    if not response.results:
        return Text("No results.", style="dim")
    blocks: list[RenderableType] = []
    for rank, hit in enumerate(response.results, start=1):
        heading = Text.assemble(
            (f"{rank}. ", "dim"),
            (hit.metadata.title, "bold"),
            (f"  {hit.score:.3f}", _score_style(hit.score)),
        )
        meta = Text(
            f"{short_id(hit.metadata.document_id)} · {hit.metadata.type}"
            f" · chunk {hit.metadata.ordinal}"
            + (f" · {hit.metadata.source}" if hit.metadata.source else ""),
            style="dim",
        )
        text = hit.text if full_text else _excerpt(hit.text)
        blocks += [heading, meta, Text(text), Text("")]
    footer = Text(
        f"{len(response.results)} results · {response.query_tokens} query tokens"
        f" · {response.latency_ms} ms",
        style="dim",
    )
    return Group(*blocks, footer)


def _score_style(score: float) -> str:
    """Cosine similarity, banded.

    The bands are a reading aid, not a threshold the service applies — nothing
    is filtered by score. They exist because a column of six-decimal floats
    tells you nothing at a glance about whether the top hit is actually close.
    """
    if score >= 0.8:
        return "bold green"
    if score >= 0.6:
        return "yellow"
    return "red"


_EXCERPT_CHARS = 320


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _EXCERPT_CHARS else flat[:_EXCERPT_CHARS].rstrip() + "…"


def ingest_result_line(source: str, result: IngestResult) -> Text:
    """One line per file, distinguishing a re-embed from a no-op re-ingest."""
    if result.unchanged:
        return Text.assemble(
            ("  unchanged  ", "dim"), (source, "dim"), (f"  {result.chunks_reused} chunks", "dim")
        )
    parts = Text.assemble(
        ("  ingested   ", "green"),
        (source, ""),
        (f"  {result.chunks_created} chunks, {result.total_tokens:,} tokens", "dim"),
    )
    if result.superseded:
        # AD-020: something with this `source` was already indexed and has just
        # been purged. Silence here would make a re-ingest that replaced a
        # document look identical to a first ingest.
        parts.append(f"  replaced {len(result.superseded)}", style="yellow")
    return parts


def stats_view(stats: Stats) -> RenderableType:
    # `expand=True` on the label/value grids: they sit inside a full-width panel,
    # and a grid sized to its content puts the widest label flush against its
    # number ("tokens stored12,345"). Expanding pushes the values to the right
    # edge, which also lines the three panels up with each other.
    corpus = Table.grid(padding=(0, 2), expand=True)
    corpus.add_column(style="dim", no_wrap=True)
    corpus.add_column(justify="right")
    corpus.add_row("documents", f"{stats.total_documents:,}")
    corpus.add_row("chunks", f"{stats.total_chunks:,}")
    corpus.add_row("tokens stored", f"{stats.total_tokens_stored:,}")
    for status, count in sorted(stats.documents_by_status.items()):
        corpus.add_row(f"  {status}", f"{count:,}")

    collections = Table(header_style="bold", expand=True)
    collections.add_column("Collection", overflow="fold")
    collections.add_column("Provider", no_wrap=True)
    collections.add_column("Model", no_wrap=True)
    collections.add_column("Dims", justify="right", no_wrap=True)
    collections.add_column("Vectors", justify="right", no_wrap=True)
    for collection in stats.collections:
        collections.add_row(
            collection.name,
            collection.provider,
            collection.model,
            str(collection.dimensions),
            # `None` is Chroma being unreachable, `0` is an empty collection.
            # Printing both as `0` would hide an outage behind an empty corpus.
            "unreachable" if collection.vectors is None else f"{collection.vectors:,}",
            style="red" if collection.vectors is None else "",
        )

    usage = Table.grid(padding=(0, 2), expand=True)
    usage.add_column(style="dim", no_wrap=True)
    usage.add_column(justify="right")
    usage.add_row(f"exact tokens ({stats.tokens.window_days}d)", f"{stats.tokens.exact_tokens:,}")
    usage.add_row("estimated tokens", f"{stats.tokens.estimated_tokens:,}")
    usage.add_row("api calls", f"{stats.tokens.api_calls:,}")

    health = Table.grid(padding=(0, 2), expand=True)
    health.add_column(style="dim", no_wrap=True)
    health.add_column(justify="right")
    health.add_row(
        "audit spill depth",
        Text(
            f"{stats.audit_spill_depth:,}",
            # Non-zero here is what `/health` reports as `degraded`: tier-1 rows
            # went to disk because Postgres was unreachable (AD-013).
            style="bold red" if stats.degraded else "green",
        ),
    )
    health.add_row(
        "telemetry dropped",
        Text(
            f"{stats.telemetry_dropped:,}", style="yellow" if stats.telemetry_dropped else "green"
        ),
    )
    health.add_row("telemetry written", f"{stats.telemetry_written:,}")
    health.add_row("telemetry queue", f"{stats.telemetry_queue_depth:,}")
    health.add_row(
        "bursts (7d)",
        Text(f"{stats.recent_bursts:,}", style="yellow" if stats.recent_bursts else "green"),
    )

    return Group(
        Panel(corpus, title="Corpus", title_align="left", border_style="cyan"),
        Panel(collections, title="Collections", title_align="left", border_style="cyan"),
        Panel(usage, title="Embedding usage", title_align="left", border_style="cyan"),
        Panel(
            health,
            title="Observability",
            title_align="left",
            border_style="red" if stats.degraded else "cyan",
        ),
    )


def config_table(values: Mapping[str, Any], *, source: str) -> RenderableType:
    # `Any` because these are settings values of mixed type on their way to
    # being printed.
    table = Table(header_style="bold", expand=False, title=source, title_justify="left")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key in sorted(values):
        value = values[key]
        table.add_row(key, "—" if value is None else str(value))
    return table


def error_text(exc: Exception) -> RenderableType:
    """What a failed command prints, on stderr.

    `detail` is the service's own sentence. `context` carries the extension
    members from the problem body — the required scope on a 403, the retry hint
    on a 429 — which are the parts that say what to do about it.
    """
    lines: list[RenderableType] = [Text(str(exc), style="bold red")]
    context: Mapping[str, Any] = getattr(exc, "context", {}) or {}
    for key, value in context.items():
        if key == "request_id":
            continue
        lines.append(Text(f"  {key}: {value}", style="dim"))
    request_id = context.get("request_id")
    if request_id:
        # The correlation handle for the tier-1 audit row (AD-013). Printed last
        # and dimmed: it means nothing to the operator until they go looking in
        # the trail, and then it is the only thing that matters.
        lines.append(Text(f"  request_id: {request_id}", style="dim"))
    return Group(*lines)


def print_error(exc: Exception) -> None:
    err_console.print(error_text(exc))


def print_all(renderables: Sequence[RenderableType]) -> None:
    for renderable in renderables:
        console.print(renderable)
