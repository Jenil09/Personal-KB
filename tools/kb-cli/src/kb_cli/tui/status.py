"""`/v1/admin/stats`, rendered with the same code `kb status` uses.

`Static` takes a Rich renderable, so `render.stats_view` is reused verbatim
rather than reimplemented in widgets. Two views of the same numbers that drift
apart is exactly the failure this avoids — and the interesting cells here
(`audit_spill_depth` non-zero, a collection reporting `unreachable`) are the
ones where a colour difference between the two would matter most.
"""

from typing import TYPE_CHECKING, ClassVar, cast

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from kb_cli.render import stats_view
from kb_client.client import KbClient
from kb_client.models import Stats

if TYPE_CHECKING:
    from kb_cli.tui.app import KbApp

__all__ = ["StatusScreen"]


class StatusScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("r", "reload", "Reload")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="status-body"):
            yield Static("Loading…", id="status-content")
        yield Footer()

    def on_mount(self) -> None:
        self._load()

    def on_screen_resume(self) -> None:
        # Refetched on every visit rather than cached with the screen. These are
        # live numbers — a spill depth from ten minutes ago is not a status.
        self._load()

    def action_reload(self) -> None:
        self._load()

    @work(exclusive=True, group="stats")
    async def _load(self) -> None:
        app = cast("KbApp", self.app)

        async def fetch(client: KbClient) -> Stats:
            return await client.stats()

        stats = await app.run_client(fetch, failure="Could not load stats")
        content = self.query_one("#status-content", Static)
        if stats is None:
            content.update("Statistics are unavailable.")
            return
        content.update(stats_view(stats))
        if stats.degraded:
            # AD-013: tier-1 rows went to the spill file because Postgres was
            # unreachable. `/health` reports `degraded` and still answers 200,
            # so this screen is where an operator actually finds out.
            app.notify(
                f"{stats.audit_spill_depth} audit records are spilled to disk.",
                title="Service degraded",
                severity="warning",
                timeout=10,
            )
