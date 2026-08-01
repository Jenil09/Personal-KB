"""The two dialogs: confirm a delete, describe a new document."""

from typing import ClassVar, NamedTuple

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

__all__ = ["ConfirmScreen", "DocumentMetadata", "MetadataScreen"]


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
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str = "", document_type: str = "note", tags: str = "") -> None:
        super().__init__()
        self._title = title
        self._type = document_type
        self._tags = tags

    def compose(self) -> ComposeResult:
        with Vertical(id="metadata-dialog"):
            yield Label("Describe this document", id="metadata-heading")
            yield Label("Title")
            yield Input(value=self._title, placeholder="Redshift Architecture", id="meta-title")
            yield Label("Type")
            yield Input(value=self._type, placeholder="note", id="meta-type")
            yield Label("Tags  [dim](comma separated)[/]")
            yield Input(value=self._tags, placeholder="ansible, hardening", id="meta-tags")
            with Grid(id="metadata-buttons"):
                yield Button("Cancel", id="meta-cancel")
                yield Button("Ingest", variant="success", id="meta-ok")

    def on_mount(self) -> None:
        self.query_one("#meta-title", Input).focus()

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
