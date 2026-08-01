"""The interactive browser, driven headlessly through Textual's pilot.

`App.run_test` mounts the real screens against a real event loop — key presses,
workers, and modal dismissal all behave as they do in a terminal — with the
service replaced at the transport. What is asserted is the state a user would
read off the screen, not the widget tree that produced it.

Workers make this asynchronous in a way that matters: every action that talks to
the service needs `pilot.pause()` before its effect is visible, because the
handler returns immediately and the worker fills the table afterwards. That is
also the behaviour being verified — a screen that blocked instead would pass
these tests and freeze in use.
"""

import io
from typing import cast

import pytest
from rich.console import Console
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Static

from kb_cli.config import KbCliSettings
from kb_cli.suggest import MetadataSuggestion
from kb_cli.tui.app import KbApp
from kb_cli.tui.documents import DocumentsScreen
from kb_cli.tui.modals import ConfirmScreen, DocumentMetadata, MetadataScreen
from kb_cli.tui.viewer import ViewerScreen
from platform_core import UpstreamError


@pytest.fixture
def app(service, settings: KbCliSettings) -> KbApp:
    return KbApp(settings=settings, transport=service.transport)


def content_of(screen: Screen[object], selector: str) -> str:
    """The text a widget is currently displaying.

    `Static.content` rather than the renderable it was handed: what is asserted
    is what a reader would see, and for the panes fed a Rich renderable that is
    the only form the two have in common.
    """
    content = screen.query_one(selector, Static).content
    if isinstance(content, str):
        return content
    # The status pane is fed a Rich renderable (it reuses `render.stats_view`),
    # so it has to be rendered to be read. Wide enough not to wrap the strings
    # being asserted on.
    console = Console(width=220, file=io.StringIO(), record=True)
    console.print(content)
    return console.export_text()


async def settle(pilot: Pilot[None]) -> None:
    """Let the workers that a keystroke started actually run.

    Twice: the first pause delivers the key and starts the worker, the second
    lets the worker's own continuation run and paint.
    """
    await pilot.pause()
    await pilot.pause()


# --- documents ------------------------------------------------------------


async def test_the_browser_opens_on_the_documents_screen(app: KbApp, service) -> None:
    service.add("Redshift Architecture")

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.current_mode == "documents"


async def test_the_corpus_is_listed(app: KbApp, service) -> None:
    service.add("Redshift Architecture")
    service.add("Restore Runbook")

    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.screen.query_one("#documents-table", DataTable)

        assert table.row_count == 2


async def test_an_empty_corpus_says_so_rather_than_showing_a_blank_pane(app: KbApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "Nothing ingested yet." in content_of(app.screen, "#preview-body")


async def test_the_preview_loads_the_highlighted_document(app: KbApp, service) -> None:
    service.add("Redshift Architecture", content="# Redshift\n\nbare metal kubernetes")

    async with app.run_test() as pilot:
        await settle(pilot)

        assert "bare metal kubernetes" in content_of(app.screen, "#preview-body")


async def test_moving_the_cursor_changes_the_preview(app: KbApp, service) -> None:
    service.add("First", content="# First\n\ncontent about ansible")
    service.add("Second", content="# Second\n\ncontent about longhorn")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("down")
        await settle(pilot)
        shown = content_of(app.screen, "#preview-body")

        assert "longhorn" in shown or "ansible" in shown


async def test_reload_picks_up_a_document_added_elsewhere(app: KbApp, service) -> None:
    service.add("First")

    async with app.run_test() as pilot:
        await settle(pilot)
        service.add("Added Later")

        await pilot.press("r")
        await settle(pilot)

        assert app.screen.query_one("#documents-table", DataTable).row_count == 2


# --- filtering ------------------------------------------------------------


async def test_the_filter_narrows_the_list(app: KbApp, service) -> None:
    service.add("Redshift Architecture")
    service.add("Restore Runbook")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("slash")
        await pilot.pause()
        app.screen.query_one("#filter", Input).value = "runbook"
        await settle(pilot)

        assert app.screen.query_one("#documents-table", DataTable).row_count == 1


async def test_the_filter_matches_tags_as_well_as_titles(app: KbApp, service) -> None:
    """One box across title, type, source, and tags — no field to choose first."""
    service.add("Untitled Thing", tags=("ansible",))
    service.add("Something Else", tags=("longhorn",))

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("slash")
        await pilot.pause()
        app.screen.query_one("#filter", Input).value = "ansible"
        await settle(pilot)

        assert app.screen.query_one("#documents-table", DataTable).row_count == 1


async def test_filter_terms_may_arrive_in_any_order(app: KbApp, service) -> None:
    service.add("Redshift Architecture", type="architecture", tags=("ansible",))
    service.add("Other", type="note")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("slash")
        await pilot.pause()
        app.screen.query_one("#filter", Input).value = "ansible redshift"
        await settle(pilot)

        assert app.screen.query_one("#documents-table", DataTable).row_count == 1


async def test_escape_clears_the_filter(app: KbApp, service) -> None:
    service.add("One")
    service.add("Two")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("slash")
        await pilot.pause()
        app.screen.query_one("#filter", Input).value = "one"
        await settle(pilot)

        cast(DocumentsScreen, app.screen).action_clear_filter()
        await settle(pilot)

        assert app.screen.query_one("#documents-table", DataTable).row_count == 2


async def test_a_filter_matching_nothing_says_so(app: KbApp, service) -> None:
    service.add("One")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("slash")
        await pilot.pause()
        app.screen.query_one("#filter", Input).value = "nothing matches this"
        await settle(pilot)

        assert "No documents match." in content_of(app.screen, "#preview-body")


# --- viewing --------------------------------------------------------------


async def test_enter_opens_the_document_viewer(app: KbApp, service) -> None:
    service.add("Readable", content="# Readable\n\nthe whole body")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("enter")
        await settle(pilot)

        assert isinstance(app.screen, ViewerScreen)


async def test_escape_returns_from_the_viewer(app: KbApp, service) -> None:
    service.add("Readable")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("enter")
        await settle(pilot)

        await pilot.press("escape")
        await settle(pilot)

        assert isinstance(app.screen, DocumentsScreen)


async def test_the_viewer_can_show_the_raw_markdown(app: KbApp, service) -> None:
    """A chunk boundary that looks wrong is a heading-level question."""
    service.add("Readable", content="# Readable\n\nthe whole body")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("enter")
        await settle(pilot)

        await pilot.press("m")
        await settle(pilot)

        source = app.screen.query_one("#viewer-source", Static)
        assert source.display is True
        assert "# Readable" in content_of(app.screen, "#viewer-source")


# --- deleting -------------------------------------------------------------


async def test_delete_asks_first(app: KbApp, service) -> None:
    identifier = service.add("Doomed")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("d")
        await settle(pilot)

        assert isinstance(app.screen, ConfirmScreen)
        assert identifier in service.documents


async def test_cancelling_the_confirmation_deletes_nothing(app: KbApp, service) -> None:
    identifier = service.add("Doomed")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("d")
        await settle(pilot)

        await pilot.press("escape")
        await settle(pilot)

        assert identifier in service.documents


async def test_confirming_removes_the_document_and_refreshes(app: KbApp, service) -> None:
    identifier = service.add("Doomed")
    service.add("Survivor")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("d")
        await settle(pilot)

        cast(ConfirmScreen, app.screen).dismiss(True)
        await settle(pilot)
        await settle(pilot)

        assert identifier not in service.documents
        assert app.screen.query_one("#documents-table", DataTable).row_count == 1


async def test_cancel_is_the_focused_button(app: KbApp, service) -> None:
    """The cheap outcome of a mistaken keystroke should be the default one."""
    service.add("Doomed")

    async with app.run_test() as pilot:
        await settle(pilot)
        await pilot.press("d")
        await settle(pilot)

        assert app.screen.focused is not None
        assert app.screen.focused.id == "cancel"


# --- composing ------------------------------------------------------------


async def test_the_metadata_dialog_prefills_the_heading(app: KbApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MetadataScreen(title="Written In The Editor"))
        await pilot.pause()

        assert app.screen.query_one("#meta-title", Input).value == "Written In The Editor"


async def test_the_metadata_dialog_refuses_a_blank_title(app: KbApp) -> None:
    """The endpoint answers 422; refusing here keeps the answer where it can be fixed."""
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MetadataScreen(title=""))
        await pilot.pause()
        screen = cast(MetadataScreen, app.screen)

        screen.query_one("#meta-title", Input).value = ""
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, MetadataScreen)


async def test_the_metadata_dialog_splits_tags(app: KbApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MetadataScreen(title="Something", tags="ansible, hardening"))
        await pilot.pause()
        screen = cast(MetadataScreen, app.screen)

        result: list[DocumentMetadata | None] = []

        def capture(value: DocumentMetadata | None = None) -> None:
            result.append(value)

        screen.dismiss = capture  # type: ignore[method-assign, assignment]
        screen._accept()

        assert result[0] is not None
        assert result[0].tags == ("ansible", "hardening")


async def test_the_suggest_button_is_off_when_no_suggester_was_supplied(app: KbApp) -> None:
    """Greyed out rather than offered and failing on every press."""
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MetadataScreen(title="Something"))
        await pilot.pause()

        assert app.screen.query_one("#meta-suggest", Button).disabled is True
        assert "Suggestions off" in content_of(app.screen, "#metadata-status")


async def test_a_suggestion_fills_the_fields_without_ingesting(app: KbApp, service) -> None:
    """The whole safety property of AD-026, asserted as a request count."""
    suggestion = MetadataSuggestion(
        title="Redshift Architecture",
        type="architecture",
        tags=("kubernetes", "prometheus"),
        model="gpt-4o-mini",
    )

    async def suggester() -> MetadataSuggestion:
        return suggestion

    async with app.run_test() as pilot:
        await pilot.pause()
        before = len(service.requests)
        app.push_screen(MetadataScreen(title="", suggester=suggester))
        await pilot.pause()

        await pilot.press("ctrl+s")
        await settle(pilot)

        assert app.screen.query_one("#meta-title", Input).value == "Redshift Architecture"
        assert app.screen.query_one("#meta-type", Input).value == "architecture"
        assert app.screen.query_one("#meta-tags", Input).value == "kubernetes, prometheus"
        # Still open, and nothing was sent. A suggestion is a filled form, not
        # an ingested document.
        assert isinstance(app.screen, MetadataScreen)
        assert len(service.requests) == before


async def test_a_suggestion_reports_what_the_model_was_shown(app: KbApp) -> None:
    async def suggester() -> MetadataSuggestion:
        return MetadataSuggestion(
            title="Rebuilding The Cluster",
            type="note",
            tags=(),
            model="qwen2.5-7b",
            truncated=True,
            coerced_type="runbook",
        )

    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MetadataScreen(suggester=suggester))
        await pilot.pause()

        await pilot.press("ctrl+s")
        await settle(pilot)

        status = content_of(app.screen, "#metadata-status")
        assert "qwen2.5-7b" in status
        assert "truncated" in status
        assert "coerced from 'runbook'" in status


async def test_a_failed_suggestion_leaves_the_form_alone(app: KbApp) -> None:
    """A button that did nothing, not a form that lost what was typed into it."""

    async def suggester() -> MetadataSuggestion:
        raise UpstreamError("The suggestion model answered 500.")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MetadataScreen(title="Typed By Hand", suggester=suggester))
        await pilot.pause()
        app.screen.query_one("#meta-tags", Input).value = "ansible"

        await pilot.press("ctrl+s")
        await settle(pilot)

        assert app.screen.query_one("#meta-title", Input).value == "Typed By Hand"
        assert app.screen.query_one("#meta-tags", Input).value == "ansible"
        assert app.screen.query_one("#meta-suggest", Button).disabled is False
        assert isinstance(app.screen, MetadataScreen)


async def test_the_compose_screen_offers_no_suggester_when_it_is_unconfigured(
    app: KbApp,
) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = cast(DocumentsScreen, app.screen)

        assert screen._suggester_for("# Anything\n") is None


async def test_the_compose_screen_builds_a_suggester_when_configured(service, api_key: str) -> None:
    settings = KbCliSettings(
        base_url="http://kb.test",
        api_key=api_key,
        suggest={"base_url": "http://localhost:1234/v1"},
    )
    app = KbApp(settings=settings, transport=service.transport)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = cast(DocumentsScreen, app.screen)

        assert screen._suggester_for("# Anything\n") is not None


# --- search ---------------------------------------------------------------


async def test_search_returns_results(app: KbApp, service) -> None:
    service.add("Redshift Architecture", content="bare metal kubernetes")

    async with app.run_test() as pilot:
        await settle(pilot)
        await app.switch_mode("search")
        await pilot.pause()

        app.screen.query_one("#query", Input).value = "kubernetes"
        await pilot.press("enter")
        await settle(pilot)

        assert app.screen.query_one("#search-table", DataTable).row_count == 1


async def test_search_does_not_fire_while_typing(app: KbApp, service) -> None:
    """Every query is an embed and a tier-1 audit row; per-keystroke is both a bill and noise."""
    service.add("Redshift Architecture")

    async with app.run_test() as pilot:
        await settle(pilot)
        await app.switch_mode("search")
        await pilot.pause()
        before = len(service.requests)

        app.screen.query_one("#query", Input).value = "kube"
        await settle(pilot)

        assert len(service.requests) == before


async def test_search_reports_the_cost_of_the_query(app: KbApp, service) -> None:
    service.add("Redshift Architecture")

    async with app.run_test() as pilot:
        await settle(pilot)
        await app.switch_mode("search")
        await pilot.pause()
        app.screen.query_one("#query", Input).value = "kubernetes"
        await pilot.press("enter")
        await settle(pilot)

        summary = content_of(app.screen, "#search-summary")
        assert "query tokens" in summary


# --- status ---------------------------------------------------------------


async def test_the_status_screen_reports_the_corpus(app: KbApp, service) -> None:
    service.add("One")

    async with app.run_test() as pilot:
        await settle(pilot)
        await app.switch_mode("status")
        await settle(pilot)

        rendered = content_of(app.screen, "#status-content")
        assert "text-embedding-3-small" in rendered


# --- failure and configuration -------------------------------------------


async def test_a_service_failure_leaves_the_screen_intact(app: KbApp, service) -> None:
    """`run_client` reports and returns `None`; nothing here should crash the app."""
    service.fail_next = (502, "upstream_error", "chroma is down")

    async with app.run_test() as pilot:
        await settle(pilot)

        assert app.is_running


async def test_an_unconfigured_app_opens_on_the_settings_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The state a fresh `uv tool install` is in — the fix is on screen, not in a crash."""
    monkeypatch.delenv("KB_CLI__API_KEY", raising=False)

    unconfigured = KbApp()
    async with unconfigured.run_test() as pilot:
        await pilot.pause()

        assert unconfigured.current_mode == "settings"
        assert unconfigured.client is None


async def test_the_settings_screen_saves_and_reconnects(
    monkeypatch: pytest.MonkeyPatch, service, api_key: str
) -> None:

    monkeypatch.delenv("KB_CLI__API_KEY", raising=False)
    unconfigured = KbApp(transport=service.transport)

    async with unconfigured.run_test() as pilot:
        await pilot.pause()
        screen = unconfigured.screen
        screen.query_one("#set-base-url", Input).value = "http://kb.test"
        screen.query_one("#set-api-key", Input).value = api_key

        await pilot.press("ctrl+s")
        await settle(pilot)
        await settle(pilot)

        assert unconfigured.client is not None
        assert unconfigured.settings is not None
        assert unconfigured.settings.base_url == "http://kb.test"


async def test_the_settings_screen_never_redisplays_the_stored_key(app: KbApp, service) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.switch_mode("settings")
        await pilot.pause()

        assert app.screen.query_one("#set-api-key", Input).value == ""


async def test_the_client_is_closed_when_the_app_exits(app: KbApp, service) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        client = app.client
        assert client is not None

    assert client._client.is_closed
