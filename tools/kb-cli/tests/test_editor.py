"""The `$EDITOR` round trip.

The editor is replaced by a small Python script that writes a known string, so
these run without a terminal and without assuming any editor is installed. The
two behaviours worth pinning are the ones that decide whether a document gets
created: a non-zero exit is a cancel, and so is an empty buffer.
"""

import os
import sys
from pathlib import Path

import pytest

from kb_cli.editor import EDITOR_ENV_VARS, edit_text, resolve_editor
from platform_core import ConfigurationError


def fake_editor(tmp_path: Path, body: str = "# Written\n\ncontent", exit_code: int = 0) -> str:
    """An editor command that overwrites the file it is given."""
    script = tmp_path / "fake_editor.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"Path(sys.argv[1]).write_text({body!r}, encoding='utf-8')\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return f"{sys.executable} {script}"


# --- resolution -----------------------------------------------------------


def test_a_configured_editor_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISUAL", "vim")

    assert resolve_editor("nvim") == "nvim"


def test_visual_is_preferred_over_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    """`EDITOR` may be a line editor; `VISUAL` is the full-screen one."""
    monkeypatch.setenv("VISUAL", "nvim")
    monkeypatch.setenv("EDITOR", "ed")

    assert resolve_editor() == "nvim"


def test_editor_is_used_when_visual_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "nano")

    assert resolve_editor() == "nano"


def test_there_is_always_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """`vi` is required to exist by POSIX; a fallback that may not is no fallback."""
    for variable in EDITOR_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)

    assert resolve_editor() == ("notepad" if os.name == "nt" else "vi")


# --- round trip -----------------------------------------------------------


def test_what_was_written_comes_back(tmp_path: Path) -> None:
    assert edit_text("", editor=fake_editor(tmp_path)) == "# Written\n\ncontent"


def test_the_initial_text_reaches_the_editor(tmp_path: Path) -> None:
    script = tmp_path / "echo_back.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\np=Path(sys.argv[1])\n"
        "p.write_text(p.read_text(encoding='utf-8').upper(), encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = edit_text("seed text", editor=f"{sys.executable} {script}")

    assert result == "SEED TEXT"


def test_a_non_zero_exit_is_a_cancel(tmp_path: Path) -> None:
    """Quitting without saving must not ingest anything."""
    assert edit_text("", editor=fake_editor(tmp_path, exit_code=1)) is None


def test_an_empty_buffer_is_a_cancel(tmp_path: Path) -> None:
    """The endpoint rejects empty content anyway; this is the earlier no."""
    assert edit_text("", editor=fake_editor(tmp_path, body="   \n\n")) is None


def test_the_temporary_file_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It held whatever was about to enter the knowledge base."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))

    edit_text("", editor=fake_editor(tmp_path))

    assert list(scratch.iterdir()) == []


def test_the_temporary_file_is_removed_even_when_the_editor_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))

    edit_text("", editor=fake_editor(tmp_path, exit_code=3))

    assert list(scratch.iterdir()) == []


def test_the_file_is_offered_to_the_editor_as_markdown(tmp_path: Path) -> None:
    """The chunker splits on Markdown structure (AD-008); so should the editor."""
    script = tmp_path / "report_suffix.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\np=Path(sys.argv[1])\n"
        "p.write_text(p.suffix, encoding='utf-8')\n",
        encoding="utf-8",
    )

    assert edit_text("", editor=f"{sys.executable} {script}") == ".md"


def test_an_editor_that_cannot_be_started_is_a_configuration_error() -> None:
    """Naming the command beats a bare `FileNotFoundError` from `subprocess`."""
    with pytest.raises(ConfigurationError, match="not-a-real-editor"):
        edit_text("", editor="/nonexistent/not-a-real-editor")


def test_a_quoted_editor_command_is_split_correctly(tmp_path: Path) -> None:
    """`EDITOR="code --wait"` is a normal thing to have configured."""
    script = tmp_path / "flagged.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[-1]).write_text(' '.join(sys.argv[1:-1]), encoding='utf-8')\n",
        encoding="utf-8",
    )

    assert edit_text("", editor=f"{sys.executable} {script} --wait") == "--wait"
