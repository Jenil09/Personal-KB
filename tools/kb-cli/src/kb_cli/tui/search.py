"""Semantic search, with the matched chunk shown next to the ranked list.

Search is submitted, never typed-ahead. Every query is an embedding call that
costs money and a tier-1 audit row (AD-013), so a `Changed` handler firing per
keystroke would embed "b", "ba", "bar", "bare"… — which is both a bill and a
trail that no longer says what anyone searched for.

The result list shows the score because it is the only thing distinguishing a
good answer from the nearest of several bad ones: nothing is filtered by score
server-side, so the top hit for a query with no real answer still comes back
looking like a hit.
"""

from typing import TYPE_CHECKING, ClassVar, cast
from uuid import UUID

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, Static

from kb_cli.render import short_id
from kb_cli.tui.viewer import ViewerScreen
from kb_client.client import KbClient
from kb_client.models import SearchHit, SearchResponse

if TYPE_CHECKING:
    from kb_cli.tui.app import KbApp

__all__ = ["SearchScreen"]

_TOP_K = 10


class SearchScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "open", "Open document"),
        Binding("ctrl+l", "focus_query", "New query"),
        Binding("escape", "focus_query", "Query", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._hits: list[SearchHit] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Ask the knowledge base…  (enter to search)", id="query")
        yield Label("", id="search-summary")
        with Horizontal(id="search-body"):
            with Vertical(id="search-left"):
                yield DataTable(id="search-table", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="search-preview"):
                yield Static("", id="hit-title")
                yield Static("", id="hit-meta")
                yield Static("", id="hit-text", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#search-table", DataTable)
        table.add_column("Score", key="score", width=7)
        table.add_column("Title", key="title")
        table.add_column("Type", key="type", width=14)
        table.add_column("Chunk", key="ordinal", width=6)
        self.query_one("#query", Input).focus()

    def action_focus_query(self) -> None:
        self.query_one("#query", Input).focus()

    @on(Input.Submitted, "#query")
    def _submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if query:
            self._search(query)

    @work(exclusive=True, group="search")
    async def _search(self, query: str) -> None:
        app = cast("KbApp", self.app)
        self.query_one("#search-summary", Label).update("searching…")

        async def run(client: KbClient) -> SearchResponse:
            return await client.search(query, top_k=_TOP_K)

        response = await app.run_client(run, failure="Search failed")
        if response is None:
            self.query_one("#search-summary", Label).update("")
            return
        self._hits = list(response.results)
        table = self.query_one("#search-table", DataTable)
        table.clear()
        for hit in self._hits:
            table.add_row(
                f"{hit.score:.3f}",
                hit.metadata.title,
                hit.metadata.type,
                str(hit.metadata.ordinal),
                key=hit.id,
            )
        self.query_one("#search-summary", Label).update(
            f"{len(self._hits)} results  ·  {response.query_tokens} query tokens  "
            f"·  {response.latency_ms} ms"
        )
        if self._hits:
            table.move_cursor(row=0)
            self._show(self._hits[0])
            table.focus()
        else:
            self._show(None)

    @on(DataTable.RowHighlighted, "#search-table")
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        for hit in self._hits:
            if hit.id == event.row_key.value:
                self._show(hit)
                return

    def _show(self, hit: SearchHit | None) -> None:
        if hit is None:
            self.query_one("#hit-title", Static).update("")
            self.query_one("#hit-meta", Static).update("")
            self.query_one("#hit-text", Static).update("No results.")
            return
        self.query_one("#hit-title", Static).update(f"[bold]{hit.metadata.title}[/]")
        self.query_one("#hit-meta", Static).update(
            f"[dim]{short_id(hit.metadata.document_id)}  ·  {hit.metadata.type}  ·  "
            f"chunk {hit.metadata.ordinal}  ·  score {hit.score:.4f}\n"
            f"{hit.metadata.source or 'no source'}  ·  "
            f"{', '.join(hit.metadata.tags) or 'no tags'}[/]"
        )
        # The chunk that actually matched, in full and unmarked-up. This is the
        # text the embedding scored, so seeing all of it is how you tell a real
        # answer from one that matched on a heading it happens to share.
        self.query_one("#hit-text", Static).update(hit.text)

    @on(DataTable.RowSelected, "#search-table")
    def _selected(self) -> None:
        self.action_open()

    def action_open(self) -> None:
        """Open the document the highlighted chunk belongs to.

        The chunk is what matched; the document is what you want to read. The
        viewer takes it from here.
        """
        table = self.query_one("#search-table", DataTable)
        if not self._hits or table.cursor_row < 0:
            return
        try:
            hit = self._hits[table.cursor_row]
        except IndexError:
            return
        self.app.push_screen(ViewerScreen(UUID(hit.metadata.document_id), hit.metadata.title))
