"""`kb` itself — exit codes, and the single place errors become messages.

This is the layer `test_cli.py` deliberately stops short of. The commands raise;
`main` is what a shell actually sees, so the contract being asserted here is
"what does `echo $?` say, and did the operator get a sentence they can act on".
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import typer

from kb_cli import __main__, cli
from kb_cli.client import KbClient
from kb_cli.config import KbCliSettings
from platform_core import NotFoundError, UpstreamError


@pytest.fixture(autouse=True)
def wired(service, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KB_CLI__BASE_URL", "http://kb.test")

    @asynccontextmanager
    async def open_client(settings: KbCliSettings) -> AsyncIterator[KbClient]:
        client = KbClient(settings, transport=service.transport)
        try:
            yield client
        finally:
            await client.aclose()

    monkeypatch.setattr(cli, "open_client", open_client)
    yield service


def run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["kb", *args])
    return __main__.main()


def test_a_successful_command_exits_zero(
    monkeypatch: pytest.MonkeyPatch, wired, api_key: str
) -> None:
    monkeypatch.setenv("KB_CLI__API_KEY", api_key)
    wired.add("One")

    assert run(monkeypatch, "list") == 0


def test_a_service_error_exits_one_and_prints_the_services_own_sentence(
    monkeypatch: pytest.MonkeyPatch, wired, capsys: pytest.CaptureFixture[str], api_key: str
) -> None:
    monkeypatch.setenv("KB_CLI__API_KEY", api_key)
    wired.fail_next = (502, "upstream_error", "chroma is unreachable")

    code = run(monkeypatch, "list")

    assert code == 1
    assert "chroma is unreachable" in capsys.readouterr().err


def test_a_failure_prints_the_request_id_for_the_audit_trail(
    monkeypatch: pytest.MonkeyPatch, wired, capsys: pytest.CaptureFixture[str], api_key: str
) -> None:
    """AD-013: the id is how a reported failure is found in `request_logs`."""
    monkeypatch.setenv("KB_CLI__API_KEY", api_key)
    wired.fail_next = (502, "upstream_error", "chroma is unreachable")

    run(monkeypatch, "list")

    assert "request_id" in capsys.readouterr().err


def test_a_missing_configuration_exits_two_and_says_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The state a fresh `uv tool install` is in, so the message is an instruction."""
    monkeypatch.delenv("KB_CLI__API_KEY", raising=False)

    code = run(monkeypatch, "list")

    captured = capsys.readouterr().err
    assert code == 2
    assert "kb config init" in captured
    assert "KB_CLI__API_KEY" in captured


def test_a_configuration_failure_is_distinguishable_from_a_service_failure(
    monkeypatch: pytest.MonkeyPatch, wired, api_key: str
) -> None:
    """Different exit codes, because a script retrying a 502 must not retry a 2."""
    monkeypatch.delenv("KB_CLI__API_KEY", raising=False)
    unconfigured = run(monkeypatch, "list")

    monkeypatch.setenv("KB_CLI__API_KEY", api_key)
    wired.fail_next = (502, "upstream_error", "down")
    failed = run(monkeypatch, "list")

    assert unconfigured != failed


def test_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    assert run(monkeypatch, "--help") == 0


def test_an_unknown_command_prints_a_usage_error_rather_than_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`standalone_mode=False` switches off Click's reporting; `main` restores it."""
    code = run(monkeypatch, "nonsense")

    assert code == 2
    assert "No such command" in capsys.readouterr().err


def test_an_aborted_prompt_exits_130(
    monkeypatch: pytest.MonkeyPatch, wired, capsys: pytest.CaptureFixture[str], api_key: str
) -> None:
    """`kb delete` with stdin closed must not traceback at the confirmation."""
    monkeypatch.setenv("KB_CLI__API_KEY", api_key)
    identifier = wired.add("Doomed")
    monkeypatch.setattr(typer, "confirm", _aborts)

    code = run(monkeypatch, "delete", identifier)

    assert code == 130
    assert identifier in wired.documents


def _aborts(*args: object, **kwargs: object) -> bool:
    """What Typer's own prompt raises on ctrl-c or a closed stdin."""
    raise typer.Abort


def test_an_interrupt_exits_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], api_key: str
) -> None:
    """What a shell expects from SIGINT; `ingest-dir` is long enough to interrupt."""
    monkeypatch.setenv("KB_CLI__API_KEY", api_key)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.app, "__call__", interrupted)
    monkeypatch.setattr(__main__, "app", interrupted)

    assert run(monkeypatch, "list") == 130


def test_the_error_hierarchy_maps_to_distinct_outcomes() -> None:
    """`main` branches on the class, so the classes must stay distinguishable."""
    assert not issubclass(NotFoundError, UpstreamError)
    assert NotFoundError.code != UpstreamError.code
