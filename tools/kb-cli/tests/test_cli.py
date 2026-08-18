"""The subcommands, driven through Typer's runner.

These assert the parts a script depends on: the exit code, the `--json` shape,
and the destructive commands refusing to act without confirmation. The prose
formatting is `test_render.py`'s problem.

`open_client` is patched rather than the settings, because the transport is the
only thing that needs replacing — the command still resolves real settings, and
a mistake in that path would still show up here.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kb_cli import cli, render
from kb_cli.config import KbCliSettings, load_config_file
from kb_cli.suggest import suggest_metadata
from kb_client.client import KbClient


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop Rich wrapping table cells at the default 80 columns.

    Under `CliRunner` there is no terminal, so Rich falls back to 80 and a
    document title long enough to be realistic gets hyphenated across two lines
    — which makes assertions here about wrapping rather than about content.
    """
    monkeypatch.setattr(render, "console", Console(width=200, soft_wrap=True))
    monkeypatch.setattr(render, "err_console", Console(width=200, soft_wrap=True, stderr=True))


@pytest.fixture(autouse=True)
def wired(service, monkeypatch: pytest.MonkeyPatch, api_key: str):
    monkeypatch.setenv("KB_CLI__API_KEY", api_key)
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


# --- list, show -----------------------------------------------------------


def test_list_prints_the_documents(runner: CliRunner, wired) -> None:
    wired.add("Redshift Architecture", type="architecture")

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0
    assert "Redshift Architecture" in result.output


def test_list_json_is_machine_readable(runner: CliRunner, wired) -> None:
    wired.add("One")

    result = runner.invoke(cli.app, ["list", "--json"])

    assert json.loads(result.output)["total"] == 1


def test_list_forwards_a_type_filter(runner: CliRunner, wired) -> None:
    wired.add("Arch", type="architecture")
    wired.add("Note", type="note")

    result = runner.invoke(cli.app, ["list", "--type", "architecture"])

    assert "Arch" in result.output
    assert "Note" not in result.output


def test_show_prints_the_content(runner: CliRunner, wired) -> None:
    identifier = wired.add("Readable", content="# Readable\n\nthe body text")

    result = runner.invoke(cli.app, ["show", identifier])

    assert "the body text" in result.output


def test_show_metadata_only_omits_the_content(runner: CliRunner, wired) -> None:
    identifier = wired.add("Readable", content="# Readable\n\nthe body text")

    result = runner.invoke(cli.app, ["show", identifier, "--metadata"])

    assert "the body text" not in result.output
    assert "indexed" in result.output


def test_a_document_can_be_named_by_id_prefix(runner: CliRunner, wired) -> None:
    identifier = wired.add("Prefixed")

    result = runner.invoke(cli.app, ["show", identifier[:8], "--json"])

    assert json.loads(result.output)["title"] == "Prefixed"


def test_an_ambiguous_prefix_is_refused(runner: CliRunner, wired) -> None:
    """Picking the first match would delete the wrong document under `kb delete`."""
    wired.add("First", document_id="aaaaaaaa-0000-0000-0000-000000000001")
    wired.add("Second", document_id="aaaaaaaa-0000-0000-0000-000000000002")

    result = runner.invoke(cli.app, ["show", "aaaaaaaa"])

    assert result.exit_code != 0
    # The message is rendered by `__main__.main`, which is what `kb` actually
    # runs; the app itself only raises. See `test_main.py`.
    assert "matches 2 documents" in str(result.exception)


def test_an_unknown_prefix_is_a_not_found(runner: CliRunner, wired) -> None:
    result = runner.invoke(cli.app, ["show", "ffffffff"])

    assert result.exit_code != 0


# --- delete ---------------------------------------------------------------


def test_delete_asks_before_removing_anything(runner: CliRunner, wired) -> None:
    identifier = wired.add("Doomed")

    result = runner.invoke(cli.app, ["delete", identifier], input="n\n")

    assert result.exit_code == 1
    assert identifier in wired.documents


def test_delete_proceeds_when_confirmed(runner: CliRunner, wired) -> None:
    identifier = wired.add("Doomed")

    result = runner.invoke(cli.app, ["delete", identifier], input="y\n")

    assert result.exit_code == 0
    assert identifier not in wired.documents


def test_delete_yes_skips_the_prompt(runner: CliRunner, wired) -> None:
    identifier = wired.add("Doomed")

    result = runner.invoke(cli.app, ["delete", identifier, "--yes"])

    assert result.exit_code == 0
    assert wired.documents == {}


def test_delete_names_what_it_is_about_to_remove(runner: CliRunner, wired) -> None:
    """A prefix is easy to get wrong and the removal is not reversible."""
    identifier = wired.add("Important Notes")

    result = runner.invoke(cli.app, ["delete", identifier[:8]], input="n\n")

    assert "Important Notes" in result.output


def test_delete_accepts_several_documents(runner: CliRunner, wired) -> None:
    first = wired.add("One")
    second = wired.add("Two")

    runner.invoke(cli.app, ["delete", first, second, "--yes"])

    assert wired.documents == {}


# --- ingest ---------------------------------------------------------------


def test_ingest_sends_one_file(runner: CliRunner, wired, tmp_path: Path) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nthe body", encoding="utf-8")

    result = runner.invoke(cli.app, ["ingest", str(document)])

    assert result.exit_code == 0
    assert [d["title"] for d in wired.documents.values()] == ["Notes"]


def test_ingest_accepts_an_explicit_title_and_tags(
    runner: CliRunner, wired, tmp_path: Path
) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Ignored\n\nbody", encoding="utf-8")

    runner.invoke(
        cli.app, ["ingest", str(document), "--title", "Chosen", "--tag", "a", "--tag", "b"]
    )

    stored = next(iter(wired.documents.values()))
    assert stored["title"] == "Chosen"
    assert stored["tags"] == ["a", "b"]


def test_ingest_dir_walks_the_tree(runner: CliRunner, wired, tmp_path: Path) -> None:
    (tmp_path / "architecture").mkdir()
    (tmp_path / "architecture" / "one.md").write_text("# One\n\nbody", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Two\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["ingest-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert {d["title"] for d in wired.documents.values()} == {"One", "Two"}
    assert {d["type"] for d in wired.documents.values()} == {"architecture", "note"}


def test_ingest_dir_dry_run_sends_nothing(runner: CliRunner, wired, tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("# One\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["ingest-dir", str(tmp_path), "--dry-run"])

    assert "would ingest" in result.output
    assert wired.documents == {}


def test_ingest_dir_reports_unchanged_files_on_a_second_run(
    runner: CliRunner, wired, tmp_path: Path
) -> None:
    """AD-008: re-running a partly failed walk must be the normal thing to do."""
    (tmp_path / "one.md").write_text("# One\n\nbody", encoding="utf-8")
    runner.invoke(cli.app, ["ingest-dir", str(tmp_path)])

    result = runner.invoke(cli.app, ["ingest-dir", str(tmp_path)])

    assert "unchanged" in result.output
    assert len(wired.documents) == 1


def test_ingest_dir_finishes_the_walk_after_a_failure(
    runner: CliRunner, wired, tmp_path: Path
) -> None:
    """One bad file must not abandon the rest, and the exit code must say so."""
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\nbody", encoding="utf-8")
    wired.fail_next = (502, "upstream_error", "provider is down")

    result = runner.invoke(cli.app, ["ingest-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert len(wired.documents) == 1
    assert "1 ingested" in result.output
    assert "1 failed" in result.output


def test_ingest_dir_skips_an_empty_file(runner: CliRunner, wired, tmp_path: Path) -> None:
    (tmp_path / "blank.md").write_text("", encoding="utf-8")
    (tmp_path / "real.md").write_text("# Real\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["ingest-dir", str(tmp_path)])

    assert "skipped" in result.output
    assert len(wired.documents) == 1


# --- search and status ----------------------------------------------------


def test_search_prints_ranked_results(runner: CliRunner, wired) -> None:
    wired.add("Architecture", content="bare metal kubernetes")

    result = runner.invoke(cli.app, ["search", "kubernetes"])

    assert result.exit_code == 0
    assert "Architecture" in result.output


def test_search_joins_unquoted_words(runner: CliRunner, wired) -> None:
    """`kb search bare metal kubernetes` is what someone types."""
    wired.add("Architecture")

    runner.invoke(cli.app, ["search", "bare", "metal", "kubernetes"])

    assert json.loads(wired.requests[-1].content)["query"] == "bare metal kubernetes"


def test_search_json_is_machine_readable(runner: CliRunner, wired) -> None:
    wired.add("Architecture")

    result = runner.invoke(cli.app, ["search", "anything", "--json"])

    assert len(json.loads(result.output)["results"]) == 1


def test_status_reports_the_corpus(runner: CliRunner, wired) -> None:
    wired.add("One")

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "documents" in result.output


def test_status_json_is_machine_readable(runner: CliRunner, wired) -> None:
    result = runner.invoke(cli.app, ["status", "--json"])

    assert json.loads(result.output)["total_documents"] == 0


# --- metadata suggestion (AD-026) -----------------------------------------


@pytest.fixture
def suggesting(model, monkeypatch: pytest.MonkeyPatch):
    """Point the suggester at the fake model, and nothing else.

    The transport is substituted the same way `wired` substitutes the service's
    — the command still resolves real settings and still runs the real prompt
    building, reduction, and validation, because those are the parts a mistake
    would hide in.
    """
    monkeypatch.setenv("KB_CLI__SUGGEST__BASE_URL", "http://model.test/v1")

    async def through_the_fake(content: str, **kwargs):
        kwargs.pop("client", None)
        async with model.client() as client:
            return await suggest_metadata(content, client=client, **kwargs)

    monkeypatch.setattr(cli, "suggest_metadata", through_the_fake)
    return model


def test_suggest_prints_a_proposal_and_ingests_nothing(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbare metal kubernetes", encoding="utf-8")

    result = runner.invoke(cli.app, ["suggest", str(document)])

    assert result.exit_code == 0
    assert "Redshift Architecture" in result.output
    assert "architecture" in result.output
    assert wired.documents == {}


def test_suggest_json_is_machine_readable(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["suggest", str(document), "--json"])

    payload = json.loads(result.output)
    assert payload["title"] == "Redshift Architecture"
    assert payload["type"] == "architecture"
    assert payload["tags"] == ["kubernetes", "observability"]


def test_suggest_says_which_setting_turns_it_on(runner: CliRunner, wired, tmp_path: Path) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["suggest", str(document)])

    # The error propagates to `__main__.main`, which is the one place that
    # prints it — so the sentence is on the exception, not in the output.
    assert result.exit_code != 0
    assert "suggest.base_url" in str(result.exception)


def test_suggest_sends_the_corpus_tags_to_the_model(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    """ "Prefer an existing tag" is only an instruction the model can follow if it has them."""
    wired.add("Existing", tags=("ansible", "wazuh"))
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    runner.invoke(cli.app, ["suggest", str(document)])

    assert "ansible, wazuh" in suggesting.requests[0]["messages"][0]["content"]


def test_ingest_suggest_prompts_before_sending(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    # Three blank lines: accept the suggested title, type, and tags.
    result = runner.invoke(cli.app, ["ingest", str(document), "--suggest"], input="\n\n\n")

    assert result.exit_code == 0
    stored = next(iter(wired.documents.values()))
    assert stored["title"] == "Redshift Architecture"
    assert stored["type"] == "architecture"
    assert stored["tags"] == ["kubernetes", "observability"]


def test_ingest_suggest_accepts_edits_at_the_prompt(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    """The suggestion is a default, not a decision."""
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    runner.invoke(
        cli.app,
        ["ingest", str(document), "--suggest"],
        input="My Own Title\nsop\nansible, security\n",
    )

    stored = next(iter(wired.documents.values()))
    assert stored["title"] == "My Own Title"
    assert stored["type"] == "sop"
    assert stored["tags"] == ["ansible", "security"]


def test_ingest_suggest_does_not_prompt_for_what_was_given_explicitly(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["ingest", str(document), "--suggest", "--title", "Decided", "--tag", "chosen"],
        input="\n",
    )

    assert result.exit_code == 0
    stored = next(iter(wired.documents.values()))
    assert stored["title"] == "Decided"
    assert stored["tags"] == ["chosen"]
    # Only the type was left to ask about.
    assert stored["type"] == "architecture"


def test_ingest_without_suggest_calls_no_model(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    """The feature is opt-in; the plain path must not have grown a dependency on it."""
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    runner.invoke(cli.app, ["ingest", str(document)])

    assert suggesting.requests == []


def test_a_failed_suggestion_still_lets_the_ingest_proceed(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    """By the time this runs the document may only exist in a closed editor buffer."""
    suggesting.status = 500
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["ingest", str(document), "--suggest"], input="\n\n\n")

    assert result.exit_code == 0
    assert "Continuing without a suggestion" in result.output
    assert next(iter(wired.documents.values()))["title"] == "Notes"


def test_a_type_outside_the_list_is_warned_about_but_accepted(
    runner: CliRunner, wired, suggesting, tmp_path: Path
) -> None:
    """`ingest-dir` still infers types from folder names, so this cannot be a refusal."""
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nbody", encoding="utf-8")

    result = runner.invoke(cli.app, ["ingest", str(document), "--suggest"], input="\nrunbook\n\n")

    assert result.exit_code == 0
    assert "outside the suggested types" in result.output
    assert next(iter(wired.documents.values()))["type"] == "runbook"


def test_new_suggest_proposes_a_title_for_an_untitled_buffer(
    runner: CliRunner, wired, suggesting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `--suggest` an untitled buffer is an error; with it, it is the point."""
    monkeypatch.setattr(cli, "edit_text", lambda *args, **kwargs: "some prose with no heading")

    result = runner.invoke(cli.app, ["new", "--suggest"], input="\n\n\n")

    assert result.exit_code == 0
    stored = next(iter(wired.documents.values()))
    assert stored["title"] == "Redshift Architecture"
    assert stored["type"] == "architecture"
    assert stored["tags"] == ["kubernetes", "observability"]


def test_new_without_suggest_still_refuses_an_untitled_buffer(
    runner: CliRunner, wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "edit_text", lambda *args, **kwargs: "no heading here")

    result = runner.invoke(cli.app, ["new"])

    assert result.exit_code != 0
    assert wired.documents == {}


# --- config ---------------------------------------------------------------


def test_config_set_and_show_round_trip(runner: CliRunner) -> None:
    runner.invoke(cli.app, ["config", "set", "base_url", "http://stored:8000"])

    result = runner.invoke(cli.app, ["config", "show"])

    assert "http://stored:8000" in result.output


def test_config_set_works_before_an_api_key_exists(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh install has no key; requiring one first is an ordering nobody knows."""
    monkeypatch.delenv("KB_CLI__API_KEY", raising=False)

    result = runner.invoke(cli.app, ["config", "set", "base_url", "http://stored:8000"])

    assert result.exit_code == 0


def test_config_show_never_prints_the_key(runner: CliRunner, api_key: str) -> None:
    runner.invoke(cli.app, ["config", "set", "api_key", "super-secret-value"])

    result = runner.invoke(cli.app, ["config", "show"])

    assert "super-secret-value" not in result.output
    assert "********" in result.output


def test_config_set_rejects_an_unknown_setting(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "set", "nonsense", "1"])

    assert result.exit_code != 0


def test_config_set_rejects_a_non_numeric_timeout(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "set", "timeout_seconds", "soon"])

    assert result.exit_code != 0


def test_a_rejected_value_never_reaches_the_file(runner: CliRunner) -> None:
    """Saving first would leave the tool unstartable, and `config set` reads the file."""
    result = runner.invoke(cli.app, ["config", "set", "timeout_seconds", "-5"])

    assert result.exit_code != 0
    # The file, not the rendered output: `config show` prints the config path,
    # and a pytest temp directory called `pytest-51` contains the string `-5`.
    assert "timeout_seconds" not in load_config_file()


def test_config_unset_removes_a_value(runner: CliRunner) -> None:
    runner.invoke(cli.app, ["config", "set", "provider", "gemini"])

    runner.invoke(cli.app, ["config", "unset", "provider"])
    result = runner.invoke(cli.app, ["config", "show"])

    assert "gemini" not in result.output


def test_config_set_expands_a_suggest_preset(runner: CliRunner) -> None:
    """Switching back to the local model must not require looking up a port."""
    runner.invoke(cli.app, ["config", "set", "suggest.base_url", "lmstudio"])

    assert load_config_file()["suggest"] == {"base_url": "http://localhost:1234/v1"}


def test_config_set_writes_into_the_nested_group(runner: CliRunner) -> None:
    runner.invoke(cli.app, ["config", "set", "suggest.base_url", "openai"])
    runner.invoke(cli.app, ["config", "set", "suggest.model", "gpt-4o-mini"])

    assert load_config_file()["suggest"] == {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    }


def test_config_unset_removes_the_group_once_it_is_empty(runner: CliRunner) -> None:
    """A leftover `{"suggest": {}}` would read as a feature that is on."""
    runner.invoke(cli.app, ["config", "set", "suggest.base_url", "lmstudio"])

    runner.invoke(cli.app, ["config", "unset", "suggest.base_url"])

    assert "suggest" not in load_config_file()


def test_config_set_rejects_a_non_numeric_content_budget(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "set", "suggest.max_content_chars", "lots"])

    assert result.exit_code != 0
    assert "whole number" in str(result.exception)


def test_config_show_never_prints_the_model_key(runner: CliRunner) -> None:
    runner.invoke(cli.app, ["config", "set", "suggest.base_url", "openai"])
    runner.invoke(cli.app, ["config", "set", "suggest.api_key", "sk-super-secret"])

    result = runner.invoke(cli.app, ["config", "show"])

    assert "sk-super-secret" not in result.output
    assert "********" in result.output


def test_config_show_says_how_to_turn_suggestion_on(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "show"])

    assert "suggest.base_url" in result.output


def test_config_path_prints_a_path(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "path"])

    assert result.output.strip().endswith("config.json")
    assert "kb-cli" in result.output
