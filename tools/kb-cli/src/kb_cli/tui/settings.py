"""The settings screen — configure an installed `kb` without touching a file.

This is what makes `uv tool install` enough on a new machine: press F4, fill in
the service URL and the API key, save, and the client is rebuilt in place. The
same values `kb config set` writes, to the same platform config file.

Two behaviours worth stating because they are not guesses:

**A field left blank is removed, not saved as empty.** An empty `provider` means
"let the service choose" (AD-006), and storing `""` would send an empty provider
name that the registry rejects with a 422.

**The stored key is never redisplayed.** The password field starts empty with a
placeholder saying whether one is stored, and leaving it empty keeps what is
already there. A screen that renders the secret back is a screen that leaks it
to whoever is standing behind you, and it buys nothing — nobody needs to read
their own API key off a form.
"""

import os
from typing import TYPE_CHECKING, Any, ClassVar, cast

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from kb_cli.config import config_path, load_config_file, save_config_file
from platform_core import ConfigurationError, PlatformError

if TYPE_CHECKING:
    from kb_cli.tui.app import KbApp

__all__ = ["SettingsScreen"]

_SECRET_PLACEHOLDER_SET = "unchanged — type to replace"
_SECRET_PLACEHOLDER_EMPTY = "required"


class SettingsScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+s", "save", "Save"),
        Binding("r", "reload", "Reload", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="settings-body"):
            yield Label("Service", classes="settings-section")
            yield Label("URL")
            yield Input(placeholder="http://localhost:8000", id="set-base-url")
            yield Label("API key")
            yield Input(password=True, id="set-api-key")
            yield Label("Embedding provider  [dim](blank for the service default)[/]")
            yield Input(placeholder="openai", id="set-provider")

            yield Label("Timeouts", classes="settings-section")
            yield Label("Request seconds")
            yield Input(placeholder="30", id="set-timeout")
            yield Label("Ingest seconds")
            yield Input(placeholder="300", id="set-ingest-timeout")

            yield Label("Editor", classes="settings-section")
            yield Label("Command  [dim](blank uses $VISUAL, then $EDITOR)[/]")
            yield Input(placeholder="nvim", id="set-editor")

            with Horizontal(id="settings-buttons"):
                yield Button("Save", variant="success", id="save")
                yield Button("Test connection", id="test")
            yield Static("", id="settings-path")
        yield Footer()

    def on_mount(self) -> None:
        self._populate()

    def on_screen_resume(self) -> None:
        self._populate()

    def action_reload(self) -> None:
        self._populate()

    def _populate(self) -> None:
        stored = load_config_file()
        self.query_one("#set-base-url", Input).value = str(stored.get("base_url") or "")
        self.query_one("#set-provider", Input).value = str(stored.get("provider") or "")
        self.query_one("#set-timeout", Input).value = _text(stored.get("timeout_seconds"))
        self.query_one("#set-ingest-timeout", Input).value = _text(
            stored.get("ingest_timeout_seconds")
        )
        self.query_one("#set-editor", Input).value = str(stored.get("editor") or "")
        key_field = self.query_one("#set-api-key", Input)
        key_field.value = ""
        key_field.placeholder = (
            _SECRET_PLACEHOLDER_SET if stored.get("api_key") else _SECRET_PLACEHOLDER_EMPTY
        )
        self.query_one("#settings-path", Static).update(f"[dim]Saved to {config_path()}[/]")

    @on(Button.Pressed, "#save")
    def _save_pressed(self) -> None:
        self.action_save()

    def action_save(self) -> None:
        stored = load_config_file()
        try:
            values = self._collect(stored)
        except ValueError as exc:
            self.notify(str(exc), title="Not saved", severity="error")
            return
        try:
            path = save_config_file(values)
        except OSError as exc:
            self.notify(str(exc), title="Could not write the config", severity="error")
            return
        self.notify(f"Saved to {path}.")
        self._reconnect()

    def _collect(self, stored: dict[str, Any]) -> dict[str, Any]:
        # `Any` because these are settings values of mixed type on their way
        # back into the JSON file.
        values = dict(stored)
        _put(values, "base_url", self.query_one("#set-base-url", Input).value)
        _put(values, "provider", self.query_one("#set-provider", Input).value)
        _put(values, "editor", self.query_one("#set-editor", Input).value)
        _put_float(values, "timeout_seconds", self.query_one("#set-timeout", Input).value)
        _put_float(
            values, "ingest_timeout_seconds", self.query_one("#set-ingest-timeout", Input).value
        )
        key = self.query_one("#set-api-key", Input).value.strip()
        if key:
            values["api_key"] = key
        if not values.get("api_key") and not os.environ.get("KB_CLI__API_KEY"):
            # The environment counts. A machine that exports `KB_CLI__API_KEY`
            # is correctly configured, and refusing to save a URL change on it
            # because this file holds no key would be wrong about its own
            # precedence rules.
            raise ValueError("An API key is required — set one here or in KB_CLI__API_KEY.")
        return values

    def _reconnect(self) -> None:
        app = cast("KbApp", self.app)

        async def rebuild() -> None:
            try:
                await app.reload_settings()
            except ConfigurationError as exc:
                app.notify(str(exc), title="Still not configured", severity="error", timeout=10)
                return
            app.notify("Reconnected. Press F1 for documents.")

        app.call_later(rebuild)

    @on(Button.Pressed, "#test")
    def _test(self) -> None:
        """Prove the URL and key work, before finding out mid-task that they do not.

        `/v1/admin/stats` rather than `/health`, deliberately: health needs no
        key and would pass with a wrong one, which is the failure this button
        exists to catch. It needs the `write` scope (AD-024), so an operator key
        that cannot reach it is a key that cannot ingest or delete either.
        """
        app = cast("KbApp", self.app)

        async def check() -> None:
            if app.client is None:
                app.notify("Save first — there is no client yet.", severity="warning")
                return
            try:
                stats = await app.client.stats()
            except PlatformError as exc:
                app.notify(str(exc), title="Connection failed", severity="error", timeout=10)
                return
            app.notify(
                f"Connected. {stats.total_documents} documents, {stats.total_chunks} chunks.",
                title="Connection OK",
            )

        app.call_later(check)


def _text(value: Any) -> str:
    # `Any` because it renders a settings value of unknown type into a field.
    return "" if value is None else str(value)


def _put(values: dict[str, Any], key: str, raw: str) -> None:
    # `Any` for the same reason as `_collect`.
    cleaned = raw.strip()
    if cleaned:
        values[key] = cleaned
    else:
        values.pop(key, None)


def _put_float(values: dict[str, Any], key: str, raw: str) -> None:
    # `Any` for the same reason as `_collect`.
    cleaned = raw.strip()
    if not cleaned:
        values.pop(key, None)
        return
    try:
        parsed = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{key.replace('_', ' ')} must be a number, not {cleaned!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{key.replace('_', ' ')} must be greater than zero.")
    values[key] = parsed
