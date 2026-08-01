"""Browse, preview, delete — the screen `kb` opens on.

The whole corpus is loaded once (PRD §12 sizes it at ~25 documents) and filtered
in memory, which is why `/` is instant and matches across title, type, source,
and tags at the same time. `_LOAD_CEILING` stops that from being a promise this
screen cannot keep if the corpus grows an order of magnitude: past it, loading
stops and the footer says so, because a browser that quietly shows a third of
the documents is worse than one that admits it.

Preview fetches the document detail per highlighted row, in an `exclusive`
worker. Holding the arrow key down therefore issues one request that survives —
the rest are cancelled as they are superseded — rather than one per row.
"""

from typing import TYPE_CHECKING, ClassVar, cast
from uuid import UUID

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, Static

from kb_cli.client import KbClient
from kb_cli.editor import edit_text
from kb_cli.models import DocumentDetail, DocumentSummary
from kb_cli.render import humanise_bytes, short_id
from kb_cli.tui.modals import ConfirmScreen, DocumentMetadata, MetadataScreen
from kb_cli.tui.viewer import ViewerScreen

if TYPE_CHECKING:
    from kb_cli.tui.app import KbApp

__all__ = ["DocumentsScreen"]

_LOAD_CEILING = 2000

_PREVIEW_CHARS = 4000


class DocumentsScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "open", "Open"),
        Binding("d,delete", "delete", "Delete"),
        Binding("n", "compose", "New"),
        Binding("r", "reload", "Reload"),
        Binding("slash", "filter", "Filter"),
        Binding("escape", "clear_filter", "Clear filter", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._documents: list[DocumentSummary] = []
        self._visible: list[DocumentSummary] = []
        self._truncated = False

    def compose(self) -> ComposeResult:
        yield Header()
        # Hidden before it is mounted, not after. Textual auto-focuses the first
        # focusable widget as the screen mounts, so a box hidden in `on_mount`
        # has already taken focus by then — every keystroke would go into an
        # invisible input and none of the bindings below would fire.
        filter_box = Input(placeholder="Filter by title, type, source, or tag…", id="filter")
        filter_box.display = False
        yield filter_box
        with Horizontal(id="documents-body"):
            with Vertical(id="documents-left"):
                yield DataTable(id="documents-table", cursor_type="row", zebra_stripes=True)
                yield Label("", id="documents-count")
            with VerticalScroll(id="documents-preview"):
                yield Static("", id="preview-title")
                yield Static("", id="preview-meta")
                yield Static("", id="preview-body", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#documents-table", DataTable)
        table.add_column("ID", key="id", width=9)
        table.add_column("Title", key="title")
        table.add_column("Type", key="type", width=14)
        table.add_column("Tags", key="tags", width=20)
        table.add_column("Chunks", key="chunks", width=7)
        table.focus()
        self._load()

    # --- loading ----------------------------------------------------------

    def action_reload(self) -> None:
        self._load()

    @work(exclusive=True, group="load")
    async def _load(self) -> None:
        app = self._kb_app()

        async def fetch(client: KbClient) -> tuple[list[DocumentSummary], bool]:
            collected: list[DocumentSummary] = []
            truncated = False
            async for page in client.iter_documents(page_size=100):
                collected.extend(page.documents)
                if len(collected) >= _LOAD_CEILING:
                    truncated = page.has_more
                    break
            return collected, truncated

        result = await app.run_client(fetch, failure="Could not list documents")
        if result is None:
            self._set_count("could not load")
            return
        self._documents, self._truncated = result
        self._apply_filter()

    # --- filtering --------------------------------------------------------

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.display = True
        box.focus()

    def action_clear_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.value = ""
        box.display = False
        self.query_one("#documents-table", DataTable).focus()
        self._apply_filter()

    @on(Input.Changed, "#filter")
    def _filter_changed(self) -> None:
        self._apply_filter()

    @on(Input.Submitted, "#filter")
    def _filter_submitted(self) -> None:
        self.query_one("#documents-table", DataTable).focus()

    def _apply_filter(self) -> None:
        needle = self.query_one("#filter", Input).value.strip().lower()
        self._visible = [document for document in self._documents if _matches(document, needle)]
        self._fill_table()

    def _fill_table(self) -> None:
        table = self.query_one("#documents-table", DataTable)
        table.clear()
        for document in self._visible:
            table.add_row(
                short_id(document.id),
                document.title,
                document.type,
                ", ".join(document.tags),
                str(document.chunk_count),
                key=str(document.id),
            )
        shown, total = len(self._visible), len(self._documents)
        suffix = "  ·  load ceiling reached, narrow with a filter" if self._truncated else ""
        self._set_count(f"{shown} of {total} documents{suffix}")
        if self._visible:
            table.move_cursor(row=0)
            self._preview(self._visible[0].id)
        else:
            self._clear_preview("No documents match." if total else "Nothing ingested yet.")

    def _set_count(self, text: str) -> None:
        self.query_one("#documents-count", Label).update(text)

    # --- preview ----------------------------------------------------------

    @on(DataTable.RowHighlighted, "#documents-table")
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key.value:
            self._preview(UUID(event.row_key.value))

    @work(exclusive=True, group="preview")
    async def _preview(self, document_id: UUID) -> None:
        app = self._kb_app()

        async def fetch(client: KbClient) -> DocumentDetail:
            return await client.get_document(document_id)

        detail = await app.run_client(fetch, failure="Could not load the preview")
        if detail is None:
            return
        self.query_one("#preview-title", Static).update(f"[bold]{detail.title}[/]")
        self.query_one("#preview-meta", Static).update(
            f"[dim]{detail.type}  ·  {detail.chunk_count} chunks  ·  "
            f"{humanise_bytes(len(detail.content.encode()))}\n"
            f"{detail.source or 'no source'}  ·  {', '.join(detail.tags) or 'no tags'}[/]"
        )
        body = detail.content
        # Truncated because this pane is a preview and the corpus averages half
        # a megabyte a file; `enter` opens the whole thing in the viewer.
        if len(body) > _PREVIEW_CHARS:
            body = body[:_PREVIEW_CHARS].rstrip() + "\n\n… press enter to read the rest"
        self.query_one("#preview-body", Static).update(body)

    def _clear_preview(self, message: str) -> None:
        self.query_one("#preview-title", Static).update("")
        self.query_one("#preview-meta", Static).update("")
        self.query_one("#preview-body", Static).update(message)

    # --- actions ----------------------------------------------------------

    @property
    def _selected(self) -> DocumentSummary | None:
        table = self.query_one("#documents-table", DataTable)
        if not self._visible or table.cursor_row < 0:
            return None
        try:
            return self._visible[table.cursor_row]
        except IndexError:
            return None

    @on(DataTable.RowSelected, "#documents-table")
    def _row_selected(self) -> None:
        self.action_open()

    def action_open(self) -> None:
        document = self._selected
        if document is not None:
            self.app.push_screen(ViewerScreen(document.id, document.title))

    def action_delete(self) -> None:
        document = self._selected
        if document is None:
            return

        # A named closure rather than a lambda: `_delete` is a worker and
        # returns a `Worker`, which is not what `push_screen` accepts from a
        # callback. Discarding it explicitly is also the honest description —
        # nothing here awaits the deletion, the reload does.
        def on_answer(confirmed: bool | None) -> None:
            if confirmed:
                self._delete(document.id)

        self.app.push_screen(
            ConfirmScreen(
                f"Delete “{document.title}”?",
                detail=(
                    f"{document.id}\n"
                    f"{document.chunk_count} chunks in {document.collection}\n\n"
                    "Removed from Postgres and Chroma. Re-ingesting costs another embed."
                ),
            ),
            callback=on_answer,
        )

    @work(group="delete")
    async def _delete(self, document_id: UUID) -> None:
        app = self._kb_app()

        async def remove(client: KbClient) -> bool:
            return await client.delete_document(document_id)

        deleted = await app.run_client(remove, failure="Delete failed")
        if deleted is None:
            return
        app.notify(
            f"Deleted {short_id(document_id)}."
            if deleted
            # The endpoint is idempotent and answers 204 either way (PRD §6.6).
            # Someone who just watched a row disappear should be told which of
            # the two happened.
            else f"{short_id(document_id)} was already gone.",
            severity="information" if deleted else "warning",
        )
        self._load()

    def action_compose(self) -> None:
        """Write a document in `$EDITOR` and ingest it.

        `App.suspend` hands the terminal back before the editor starts and takes
        it again afterwards. Without it the editor and Textual both drive the
        same terminal at once and neither survives.
        """
        app = self._kb_app()
        with app.suspend():
            content = edit_text("# \n\n", editor=app.settings.editor if app.settings else None)
        if content is None:
            app.notify("Nothing ingested.", severity="information")
            return

        def on_described(metadata: DocumentMetadata | None) -> None:
            if metadata is not None:
                self._ingest(content, metadata)

        app.push_screen(MetadataScreen(title=_heading_of(content)), callback=on_described)

    @work(group="ingest")
    async def _ingest(self, content: str, metadata: DocumentMetadata) -> None:
        app = self._kb_app()

        async def send(client: KbClient) -> object:
            return await client.ingest(
                title=metadata.title,
                content=content,
                document_type=metadata.type,
                # No `source`: this did not come from a file, and inventing one
                # would make a later ingest of a real file with that name
                # supersede it (AD-020).
                source=None,
                tags=metadata.tags,
            )

        result = await app.run_client(send, failure="Ingest failed")
        if result is None:
            return
        app.notify(f"Ingested “{metadata.title}”.")
        self._load()

    def _kb_app(self) -> "KbApp":
        # `KbApp` imports this screen to build its mode table, so the reference
        # back is a type-checking one and the cast is unchecked at runtime. The
        # screen only ever runs inside the app that mounted it.
        return cast("KbApp", self.app)


def _matches(document: DocumentSummary, needle: str) -> bool:
    if not needle:
        return True
    haystack = " ".join(
        (document.title, document.type, document.source or "", " ".join(document.tags))
    ).lower()
    # Every whitespace-separated term must appear somewhere, in any order:
    # `arch ansible` finds the architecture document tagged ansible without
    # needing to know which field holds which.
    return all(term in haystack for term in needle.split())


def _heading_of(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            return ""
    return ""
