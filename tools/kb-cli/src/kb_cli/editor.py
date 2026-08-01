"""Round-tripping text through the operator's own editor.

Used by the compose-a-document action, which exists because the most common
thing to ingest is something not yet written down anywhere: writing it to a file
first, ingesting the file, then deleting the file is three steps to avoid one
`$EDITOR`.

The temporary file is a `.md` because that is what the chunker splits on
(AD-008) and because syntax highlighting makes the difference between writing
Markdown and typing into a beige box.
"""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from platform_core import ConfigurationError

__all__ = ["EDITOR_ENV_VARS", "edit_text", "resolve_editor"]

EDITOR_ENV_VARS = ("VISUAL", "EDITOR")
"""Checked in order, which is the convention every other tool follows.

`VISUAL` first: it is the full-screen editor, and `EDITOR` is historically
allowed to be a line editor. On a machine where they differ, the one the user
meant for this is `VISUAL`.
"""


def resolve_editor(configured: str | None = None) -> str:
    """The editor command, as a string still to be split by `shlex`.

    Falls back to `notepad` on Windows and `vi` elsewhere. `vi` rather than
    `nano` because POSIX requires it to exist; a fallback that may not be
    installed is not a fallback.
    """
    if configured:
        return configured
    for variable in EDITOR_ENV_VARS:
        value = os.environ.get(variable)
        if value:
            return value
    return "notepad" if os.name == "nt" else "vi"


def edit_text(
    initial: str = "",
    *,
    editor: str | None = None,
    suffix: str = ".md",
) -> str | None:
    """Open `initial` in the editor; return what came back, or `None`.

    `None` means "do not proceed" and covers both ways of saying it — the editor
    exiting non-zero (`:cq`), and a buffer saved with nothing but whitespace in
    it. Quitting without saving is how you cancel, so it must not ingest an
    empty document, and an empty document would be rejected by the endpoint
    anyway.

    The file is deleted in a `finally`: it holds whatever was about to enter the
    knowledge base, and leaving that in the system temporary directory outlives
    the reason it was there.
    """
    command = shlex.split(resolve_editor(editor), posix=os.name != "nt")
    if not command:
        raise ConfigurationError("No editor is configured and none could be inferred.")

    handle, name = tempfile.mkstemp(suffix=suffix, prefix="kb-")
    path = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(initial)
        try:
            completed = subprocess.run([*command, str(path)], check=False)
        except OSError as exc:
            raise ConfigurationError(
                f"Could not start the editor {command[0]!r}: {exc}",
                context={"editor": command[0]},
            ) from exc
        if completed.returncode != 0:
            return None
        edited = path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)
    return edited if edited.strip() else None
