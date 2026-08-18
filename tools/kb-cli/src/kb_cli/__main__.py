"""`kb` — the console script, and the one place a failure becomes an exit code.

The same argument as the service having exactly one exception handler: commands
raise, this translates. A `PlatformError` prints the service's own sentence and
exits `1`; anything unexpected keeps its traceback, because an unexpected
exception in an operator tool is a bug report and swallowing it helps nobody.

`ConfigurationError` is separated out because on a fresh `uv tool install` it is
not an error so much as the tool's first instruction — the message names the
command that fixes it rather than the pydantic field that raised it.
"""

import sys
from typing import TYPE_CHECKING

import typer

from kb_cli import render
from kb_cli.cli import app
from kb_cli.config import CONFIG_PATH_ENV_VAR, config_path
from platform_core import ConfigurationError, PlatformError

if TYPE_CHECKING:
    # For the annotation below only. The runtime import is the guarded one
    # inside `_usage_error_types`; naming the class here is what lets a checker
    # see `.show()` and `.exit_code` on the exception the handler catches.
    from typer._click.exceptions import ClickException

__all__ = ["main"]


def _usage_error_types() -> tuple[type["ClickException"], ...]:
    """`ClickException`, from wherever this Typer keeps it.

    Typer 0.27 vendors its own copy of Click and does **not** depend on the
    `click` distribution. So `typer._click.exceptions.ClickException` is not the
    same class object as `click.exceptions.ClickException`, and on a clean
    `uv tool install` the latter does not exist at all — importing it
    unconditionally makes `kb` fail at startup with `ModuleNotFoundError`, which
    is how this was found. The private module is imported first because it is
    the one that will actually be raised; the public one is the fallback for a
    future Typer that stops vendoring.
    """
    try:
        from typer._click.exceptions import ClickException
    except ImportError:  # pragma: no cover - Typer without a vendored Click
        from click.exceptions import ClickException  # type: ignore[assignment]
    return (ClickException,)


_USAGE_ERRORS = _usage_error_types()

# `typer.Abort` is public and is the vendored class, so it needs no such dance.
_ABORTS = (typer.Abort,)


def main() -> int:
    try:
        app(standalone_mode=False)
    except SystemExit as exit_signal:
        return int(exit_signal.code or 0)
    except _USAGE_ERRORS as exc:
        # `standalone_mode=False` is what lets a `PlatformError` reach the
        # handlers below, but it also switches off Click's own reporting — so a
        # mistyped command would otherwise print a traceback instead of "No such
        # command". Click already knows how to say this, and its exit code (2
        # for a usage error) is the one shells expect.
        exc.show()
        return int(exc.exit_code)
    except _ABORTS:
        # A confirmation answered with ctrl-c or EOF — `kb delete` without
        # `--yes` under a closed stdin.
        render.err_console.print("Aborted.", style="dim")
        return 130
    except ConfigurationError as exc:
        render.print_error(exc)
        render.err_console.print(
            f"\nConfigure this machine with [bold]kb config init[/], "
            f"which writes {config_path()}.\n"
            f"Environment variables ([bold]KB_CLI__BASE_URL[/], [bold]KB_CLI__API_KEY[/]) "
            f"override it, and [bold]{CONFIG_PATH_ENV_VAR}[/] points it elsewhere.",
            style="dim",
        )
        return 2
    except PlatformError as exc:
        render.print_error(exc)
        return 1
    except KeyboardInterrupt:
        # 130 is what a shell expects from SIGINT, and `ingest-dir` is long
        # enough that interrupting it is a normal thing to do rather than a
        # failure worth a traceback.
        render.err_console.print("Interrupted.", style="dim")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
