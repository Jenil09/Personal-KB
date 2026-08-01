"""The two dialogs: confirm a delete, describe a new document."""

from collections.abc import Awaitable, Callable
from typing import ClassVar, NamedTuple

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from kb_cli.suggest import MetadataSuggestion
from platform_core import PlatformError

__all__ = ["ConfirmScreen", "DocumentMetadata", "MetadataScreen", "Suggester"]

Suggester = Callable[[], Awaitable[MetadataSuggestion]]
"""Everything the dialog needs to know about the LLM: awaiting it yields a
suggestion. The content, the settings, and the tag vocabulary are bound by
whoever opened the screen, which keeps `httpx` and configuration out of a module
whose job is drawing two boxes — and lets the tests hand it a coroutine."""


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no the operator has to answer deliberately.

    Cancel is the focused button and `escape` dismisses as `False`, because the
    only thing this guards is a delete: the vectors go, and re-ingesting costs
    another embed. The cheap outcome of a mistaken keystroke should be the
    default one.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss_false", "Cancel", show=False)
    ]

    def __init__(self, question: str, detail: str = "", confirm_label: str = "Delete") -> None:
        super().__init__()
        self._question = question
        self._detail = detail
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Grid(id="confirm-grid"):
            yield Label(self._question, id="confirm-question")
            yield Static(self._detail, id="confirm-detail")
            yield Button("Cancel", variant="primary", id="cancel")
            yield Button(self._confirm_label, variant="error", id="confirm")

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    @on(Button.Pressed, "#confirm")
    def _confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(False)

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class DocumentMetadata(NamedTuple):
    title: str
    type: str
    tags: tuple[str, ...]


class MetadataScreen(ModalScreen[DocumentMetadata | None]):
    """Title, type, and tags for a document composed in the editor.

    Asked *after* the editor rather than before, so the title can be prefilled
    from the `# ` heading that was just written — which is what makes the common
    case a single `enter`.

    `ctrl+s` fills the three fields from an LLM (AD-026) when a suggester was
    supplied. **It fills them and stops there.** This screen never dismisses
    itself on a suggestion and never constructs a `DocumentMetadata` from one:
    the only path to ingest is `_accept` reading back whatever is in the inputs
    after a human has looked at them.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+s", "suggest", "Suggest metadata", show=False),
    ]

    def __init__(
        self,
        title: str = "",
        document_type: str = "note",
        tags: str = "",
        *,
        suggester: Suggester | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._type = document_type
        self._tags = tags
        self._suggester = suggester

    def compose(self) -> ComposeResult:
        with Vertical(id="metadata-dialog"):
            yield Label("Describe this document", id="metadata-heading")
            yield Label("Title")
            yield Input(value=self._title, placeholder="Redshift Architecture", id="meta-title")
            yield Label("Type")
            yield Input(value=self._type, placeholder="note", id="meta-type")
            yield Label("Tags  [dim](comma separated)[/]")
            yield Input(value=self._tags, placeholder="ansible, hardening", id="meta-tags")
            yield Static(self._initial_status(), id="metadata-status")
            with Grid(id="metadata-buttons"):
                yield Button("Cancel", id="meta-cancel")
                yield Button("Suggest", id="meta-suggest", disabled=self._suggester is None)
                yield Button("Ingest", variant="success", id="meta-ok")

    def _initial_status(self) -> str:
        if self._suggester is None:
            return "[dim]Suggestions off — configure `suggest.base_url` to enable them.[/]"
        return "[dim]ctrl+s suggests a title, type, and tags.[/]"

    def on_mount(self) -> None:
        self.query_one("#meta-title", Input).focus()

    # --- suggestion -------------------------------------------------------

    @on(Button.Pressed, "#meta-suggest")
    def _suggest_pressed(self) -> None:
        self.action_suggest()

    def action_suggest(self) -> None:
        if self._suggester is not None:
            self._suggest()

    @work(exclusive=True, group="suggest")
    async def _suggest(self) -> None:
        suggester = self._suggester
        if suggester is None:
            return
        button = self.query_one("#meta-suggest", Button)
        status = self.query_one("#metadata-status", Static)
        button.disabled = True
        status.update("[dim]Asking the model…[/]")
        try:
            suggestion = await suggester()
        except PlatformError as exc:
            # Every field is left exactly as it was. A failed suggestion is a
            # button that did nothing, not a form that lost what was typed in
            # it.
            status.update("[dim]ctrl+s suggests a title, type, and tags.[/]")
            self.notify(str(exc), title="Could not suggest metadata", severity="error", timeout=10)
            return
        finally:
            button.disabled = False
        self._apply(suggestion)

    def _apply(self, suggestion: MetadataSuggestion) -> None:
        self.query_one("#meta-title", Input).value = suggestion.title
        self.query_one("#meta-type", Input).value = suggestion.type
        self.query_one("#meta-tags", Input).value = ", ".join(suggestion.tags)
        self.query_one("#metadata-status", Static).update(
            "[dim]" + "  ·  ".join(suggestion.notes) + "  ·  edit anything before ingesting[/]"
        )

    @on(Button.Pressed, "#meta-ok")
    @on(Input.Submitted)
    def _accept(self) -> None:
        title = self.query_one("#meta-title", Input).value.strip()
        document_type = self.query_one("#meta-type", Input).value.strip() or "note"
        if not title:
            # The endpoint rejects a blank title with a 422. Refusing here keeps
            # the answer in the dialog that can still fix it.
            self.notify("A title is required.", severity="warning")
            self.query_one("#meta-title", Input).focus()
            return
        raw = self.query_one("#meta-tags", Input).value
        tags = tuple(tag.strip() for tag in raw.split(",") if tag.strip())
        self.dismiss(DocumentMetadata(title=title, type=document_type, tags=tags))

    @on(Button.Pressed, "#meta-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
