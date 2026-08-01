"""Reading one document in full.

Pushed on top of whichever mode asked for it, so `escape` returns to the list
with its cursor where it was.

The content is rendered as Markdown because that is what the corpus is and what
the chunker splits on (AD-008) — but through `Markdown.update` on a plain
string, not `Markdown(path)`: the text came from the service, not from a file,
and Textual's Markdown widget resolves relative links against a document path it
would not have.
"""

from typing import TYPE_CHECKING, ClassVar, cast
from uuid import UUID

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, LoadingIndicator, Markdown, Static

from kb_cli.client import KbClient
from kb_cli.models import DocumentDetail
from kb_cli.render import humanise_bytes

if TYPE_CHECKING:
    from kb_cli.tui.app import KbApp

__all__ = ["ViewerScreen"]


class ViewerScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,q", "app.pop_screen", "Back"),
        Binding("r", "reload", "Reload"),
        # `m` toggles raw Markdown source. Worth having: a chunk boundary that
        # looks wrong is a heading-level question, and rendered output is
        # exactly where heading levels stop being visible.
        Binding("m", "toggle_source", "Source"),
    ]

    def __init__(self, document_id: UUID, title: str = "") -> None:
        super().__init__()
        self._document_id = document_id
        self._title = title
        self._detail: DocumentDetail | None = None
        self._show_source = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(self._title or str(self._document_id), id="viewer-title")
        yield Static("", id="viewer-meta")
        with VerticalScroll(id="viewer-body"):
            yield LoadingIndicator(id="viewer-loading")
            yield Markdown("", id="viewer-markdown")
            yield Static("", id="viewer-source")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#viewer-markdown", Markdown).display = False
        self.query_one("#viewer-source", Static).display = False
        self._load()

    def action_reload(self) -> None:
        self._load()

    def action_toggle_source(self) -> None:
        self._show_source = not self._show_source
        if self._detail is not None:
            self._display(self._detail)

    @work(exclusive=True)
    async def _load(self) -> None:
        # Type-checking reference only: the app imports this screen, so a
        # runtime import back would close the cycle for no benefit.
        app = cast("KbApp", self.app)

        async def fetch(client: KbClient) -> DocumentDetail:
            return await client.get_document(self._document_id)

        detail = await app.run_client(fetch, failure="Could not load the document")
        self.query_one("#viewer-loading", LoadingIndicator).display = False
        if detail is None:
            return
        self._detail = detail
        self._display(detail)

    def _display(self, detail: DocumentDetail) -> None:
        self.query_one("#viewer-title", Label).update(detail.title)
        self.query_one("#viewer-meta", Static).update(
            f"[dim]{detail.id}  ·  {detail.type}  ·  {detail.chunk_count} chunks  ·  "
            f"{humanise_bytes(len(detail.content.encode()))}  ·  "
            f"{', '.join(detail.tags) or 'no tags'}[/]"
        )
        markdown = self.query_one("#viewer-markdown", Markdown)
        source = self.query_one("#viewer-source", Static)
        markdown.display = not self._show_source
        source.display = self._show_source
        if self._show_source:
            # `Static` with markup off: this is the document's own text, and
            # letting Rich interpret a literal `[dim]` in someone's notes as a
            # style tag would silently eat it.
            source.update(detail.content)
        else:
            self.call_next(markdown.update, detail.content)

    @on(Markdown.LinkClicked)
    def _ignore_link(self, event: Markdown.LinkClicked) -> None:
        # Links point at the author's own filesystem or the open web; this
        # viewer resolves neither. Saying so beats a click that does nothing.
        event.prevent_default()
        self.notify(f"Links are not followed here: {event.href}", severity="information")
