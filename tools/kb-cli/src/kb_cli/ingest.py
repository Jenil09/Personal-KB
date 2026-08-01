"""Turning files on disk into ingest requests.

Pure functions over paths and text, with no client and no I/O beyond reading the
files themselves — which is what makes the interesting parts (which files are
picked, what a document ends up titled, what `source` it claims) testable
without a service, and is where the unit tests are.

The one rule everything here serves: **`source` is the identity of a document**.
AD-020 supersedes on `source`, so re-ingesting a directory has to produce the
same `source` string it produced last time or every file lands twice. That is
why it is the path relative to the walk root in POSIX form, and never the
absolute path — an absolute path embeds the machine, and ingesting the same
folder from a laptop and from the VPS would duplicate the entire corpus.
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

__all__ = [
    "DEFAULT_EXCLUDES",
    "DEFAULT_GLOBS",
    "LocalDocument",
    "SkippedFile",
    "discover",
    "read_document",
    "read_documents",
    "source_for",
]

DEFAULT_GLOBS: tuple[str, ...] = ("*.md", "*.markdown", "*.txt", "*.rst")
"""Text formats the chunker understands (AD-008).

Not `*` with a binary sniff: a directory walk that silently decides a PDF is
ingestable produces a document full of mojibake, embeds it, and charges for the
privilege. Widening this is a deliberate `--glob`.
"""

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)

_H1 = re.compile(r"^\s{0,3}#\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)

_MAX_TITLE_CHARS = 512


@dataclass(frozen=True, slots=True)
class LocalDocument:
    """One file, ready to become an ingest request."""

    path: Path
    title: str
    content: str
    source: str
    type: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A file the walk found and deliberately did not ingest."""

    path: Path
    reason: str


@dataclass(slots=True)
class ReadResult:
    documents: list[LocalDocument] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)


def discover(
    root: Path,
    *,
    globs: Sequence[str] = DEFAULT_GLOBS,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    recursive: bool = True,
) -> list[Path]:
    """Every matching file under `root`, sorted, de-duplicated.

    Sorted because the order files are ingested in is visible: it is the order
    progress is reported in, and — when a run is interrupted and restarted — the
    prefix that is already done. `set` first because overlapping globs (`*.md`
    and `*.markdown` do not overlap, but `*.md` and `notes*` do) would otherwise
    ingest a file twice in one run.
    """
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    if root.is_file():
        return [root]
    excluded = set(excludes)
    found: set[Path] = set()
    for pattern in globs:
        matches = root.rglob(pattern) if recursive else root.glob(pattern)
        found.update(
            path
            for path in matches
            if path.is_file()
            # Any excluded component, at any depth: `.git/objects/…` must go,
            # not just a file literally named `.git`.
            and not (excluded & set(path.relative_to(root).parts))
        )
    return sorted(found)


def source_for(path: Path, root: Path) -> str:
    """The stable `source` for a file — see this module's docstring."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        # `path` is not under `root`, which happens when a single file is
        # ingested by name. Its own name is the most stable thing available.
        relative = Path(path.name)
    return str(PurePosixPath(*relative.parts))


def title_for(path: Path, content: str) -> str:
    """The first Markdown H1, or the filename made presentable.

    The heading wins because it is what the author wrote; the filename is a
    fallback that has to survive `redshift_architecture.md` and
    `2026-07-30-notes.txt` without looking like a filename in a document list.
    """
    match = _H1.search(content)
    if match:
        title = " ".join(match.group("title").split())
        if title:
            return title[:_MAX_TITLE_CHARS]
    stem = unicodedata.normalize("NFC", path.stem)
    words = [word for word in re.split(r"[-_\s]+", stem) if word]
    return (" ".join(words) or path.name)[:_MAX_TITLE_CHARS]


def type_for(path: Path, root: Path, override: str | None = None) -> str:
    """`--type` if given, else the directory the file sits in, else `note`.

    The directory is a reasonable guess precisely because it is how a knowledge
    base tends to already be organised — `architecture/`, `runbooks/`, `notes/`
    are the types someone would have typed in by hand. Files directly under the
    root have no such signal and get `note` rather than the root's own name,
    which would be the name of whatever folder the corpus happens to live in.
    """
    if override:
        return override
    parent = PurePosixPath(source_for(path, root)).parent
    if str(parent) == ".":
        return "note"
    return parent.parts[-1]


def read_document(
    path: Path,
    root: Path,
    *,
    document_type: str | None = None,
    tags: Sequence[str] = (),
) -> LocalDocument:
    """Read and describe one file. Raises `ValueError` if it is not ingestable."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("not valid UTF-8") from exc
    if not content.strip():
        # The endpoint rejects empty content with a 422. Catching it here means
        # an empty file in a 40-file walk is a skip line rather than a failure
        # that stops the run.
        raise ValueError("empty")
    return LocalDocument(
        path=path,
        title=title_for(path, content),
        content=content,
        source=source_for(path, root),
        type=type_for(path, root, document_type),
        tags=tuple(tags),
    )


def read_documents(
    paths: Iterable[Path],
    root: Path,
    *,
    document_type: str | None = None,
    tags: Sequence[str] = (),
) -> ReadResult:
    """Read many, collecting the unreadable ones rather than raising."""
    result = ReadResult()
    for path in paths:
        try:
            result.documents.append(
                read_document(path, root, document_type=document_type, tags=tags)
            )
        except (ValueError, OSError) as exc:
            result.skipped.append(SkippedFile(path=path, reason=str(exc)))
    return result
