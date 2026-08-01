"""The Textual application — one client, four modes, one place errors surface.

Modes rather than a stack of pushed screens, because the four top-level views
are peers: Documents, Search, Status, Settings. Each keeps its own state when
you leave it, so tabbing to Status mid-search and back does not throw the search
away. Screens that *are* transient — the document viewer, the confirmations —
are pushed on top of whichever mode is active.

**The client is owned here, not by the screens.** One `httpx.AsyncClient` per
process is the rule the service follows for its own outbound calls, and it is
what lets the settings screen rebuild the connection in place after saving a new
URL — a screen holding its own client would leave four of them pointed at the
old one.

**Nothing here catches an error to hide it.** `run_client` funnels every call
through one place that turns a `PlatformError` into a toast carrying the
service's own sentence, so a 403 in the browser says the same thing it says in
`kb list`.
"""

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, TypeVar

import httpx
from textual.app import App
from textual.binding import Binding, BindingType
from textual.screen import Screen

from kb_cli.client import KbClient
from kb_cli.config import KbCliSettings, load_settings
from kb_cli.tui.documents import DocumentsScreen
from kb_cli.tui.search import SearchScreen
from kb_cli.tui.settings import SettingsScreen
from kb_cli.tui.status import StatusScreen
from platform_core import ConfigurationError, PlatformError

__all__ = ["KbApp"]

T = TypeVar("T")


class KbApp(App[None]):
    """`kb` with no arguments."""

    CSS_PATH = "kb.tcss"
    TITLE = "Personal Knowledge Base"

    MODES: ClassVar[dict[str, str | Callable[[], Screen[Any]]]] = {
        "documents": DocumentsScreen,
        "search": SearchScreen,
        "status": StatusScreen,
        "settings": SettingsScreen,
    }

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("f1", "switch_mode('documents')", "Documents"),
        Binding("f2", "switch_mode('search')", "Search"),
        Binding("f3", "switch_mode('status')", "Status"),
        Binding("f4", "switch_mode('settings')", "Settings"),
        # `ctrl+q` rather than a bare `q`: this application has text inputs on
        # three of its four screens, and a single-letter quit that fires while
        # someone is typing a query is the fastest way to make a tool feel
        # hostile.
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        settings: KbCliSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        # Injected by the tests, which drive the screens against a transport
        # serving the `/v1` contract. Nothing else in this application takes a
        # transport, which is what keeps the seam to one constructor argument.
        self._transport = transport
        self._settings = settings
        self._client: KbClient | None = None
        self._settings_error: ConfigurationError | None = None
        if settings is None:
            try:
                self._settings = load_settings()
            except ConfigurationError as exc:
                # A fresh `uv tool install` has no config file and no API key.
                # Failing to start would be correct and useless; opening on the
                # settings screen with the reason showing is the same
                # information plus the way to fix it.
                self._settings_error = exc

    @property
    def settings(self) -> KbCliSettings | None:
        return self._settings

    @property
    def client(self) -> KbClient | None:
        return self._client

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def on_mount(self) -> None:
        self._open_client()
        if self._settings_error is not None:
            await self.switch_mode("settings")
            self.notify(
                str(self._settings_error), title="Not configured", severity="warning", timeout=10
            )
        else:
            await self.switch_mode("documents")

    async def on_unmount(self) -> None:
        await self._close_client()

    def _open_client(self) -> None:
        if self._settings is not None:
            self._client = KbClient(self._settings, transport=self._transport)

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def reload_settings(self) -> None:
        """Re-read configuration and rebuild the client in place.

        Called by the settings screen after a save. The old client is closed
        first — an `AsyncClient` that goes out of scope with its connection pool
        open is exactly the resource leak the lifespan-owned-client rule exists
        to prevent, and here it would accumulate one per save.
        """
        await self._close_client()
        self._settings_error = None
        try:
            self._settings = load_settings()
        except ConfigurationError as exc:
            self._settings_error = exc
            self._settings = None
            raise
        self._open_client()

    async def run_client(
        self, work: Callable[[KbClient], Awaitable[T]], *, failure: str = "Request failed"
    ) -> T | None:
        """Run one call against the client, reporting failure as a toast.

        Returns `None` on failure, which every caller treats as "leave the view
        as it was". That is deliberately the same answer as an unconfigured
        client: a screen has nothing useful to draw in either case, and the
        toast has already said which it was.
        """
        if self._client is None:
            self.notify(
                "No service is configured. Press F4 to set one up.",
                title="Not configured",
                severity="warning",
            )
            return None
        try:
            return await work(self._client)
        except PlatformError as exc:
            self.notify(str(exc), title=failure, severity="error", timeout=10)
            return None
