"""Which files get picked, and what they claim to be.

The centre of gravity is `source_for`. AD-020 supersedes on `source`, so the
question "does re-ingesting this directory replace the documents or duplicate
them" is answered entirely by whether this function is stable — across working
directories, across absolute and relative arguments, and across platforms.
"""

from pathlib import Path

import pytest

from kb_cli.ingest import (
    DEFAULT_EXCLUDES,
    DEFAULT_GLOBS,
    discover,
    read_document,
    read_documents,
    source_for,
    title_for,
    type_for,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "architecture").mkdir()
    (tmp_path / "runbooks").mkdir()
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / "top.md").write_text("# Top Level\n\nbody", encoding="utf-8")
    (tmp_path / "architecture" / "redshift.md").write_text("# Redshift\n\nbody", encoding="utf-8")
    (tmp_path / "runbooks" / "restore.txt").write_text("restore steps", encoding="utf-8")
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4 binary")
    (tmp_path / ".git" / "objects" / "cached.md").write_text("# Not mine", encoding="utf-8")
    return tmp_path


# --- discovery ------------------------------------------------------------


def test_discover_finds_matching_files_recursively(corpus: Path) -> None:
    found = discover(corpus)

    assert [path.name for path in found] == ["redshift.md", "restore.txt", "top.md"]


def test_discover_skips_excluded_directories_at_any_depth(corpus: Path) -> None:
    """`.git/objects/cached.md` is nested; excluding only top-level names misses it."""
    assert all(".git" not in path.parts for path in discover(corpus))


def test_discover_ignores_formats_outside_the_globs(corpus: Path) -> None:
    """A PDF read as UTF-8 becomes an embedded document full of mojibake."""
    assert all(path.suffix != ".pdf" for path in discover(corpus))


def test_discover_is_sorted_and_deduplicated(corpus: Path) -> None:
    """Overlapping globs must not ingest one file twice in a single run."""
    found = discover(corpus, globs=("*.md", "*.md", "top*"))

    assert found == sorted(found)
    assert len(found) == len(set(found))


def test_discover_can_stay_shallow(corpus: Path) -> None:
    assert [path.name for path in discover(corpus, recursive=False)] == ["top.md"]


def test_discover_accepts_a_single_file(corpus: Path) -> None:
    assert discover(corpus / "top.md") == [corpus / "top.md"]


def test_discover_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nowhere")


def test_the_default_excludes_cover_the_usual_noise() -> None:
    assert {".git", ".venv", "node_modules", "__pycache__"} <= set(DEFAULT_EXCLUDES)


def test_the_default_globs_are_text_formats() -> None:
    assert all(pattern.startswith("*.") for pattern in DEFAULT_GLOBS)


# --- source ---------------------------------------------------------------


def test_source_is_relative_to_the_walk_root(corpus: Path) -> None:
    assert source_for(corpus / "architecture" / "redshift.md", corpus) == "architecture/redshift.md"


def test_source_is_posix_regardless_of_platform(corpus: Path) -> None:
    """A Windows-shaped `source` would not match the same file ingested on Linux."""
    assert "\\" not in source_for(corpus / "architecture" / "redshift.md", corpus)


def test_source_is_identical_for_relative_and_absolute_roots(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-ingesting from a different working directory must supersede, not duplicate."""
    absolute = source_for(corpus / "architecture" / "redshift.md", corpus)

    monkeypatch.chdir(corpus)
    relative = source_for(Path("architecture/redshift.md"), Path())

    assert absolute == relative


def test_source_of_a_file_outside_the_root_falls_back_to_its_name(
    corpus: Path, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "stray.md"

    assert source_for(outside, corpus) == "stray.md"


# --- title and type -------------------------------------------------------


def test_title_prefers_the_first_h1() -> None:
    assert title_for(Path("whatever.md"), "# Redshift Architecture\n\nbody") == (
        "Redshift Architecture"
    )


def test_title_ignores_deeper_headings() -> None:
    assert title_for(Path("my_notes.md"), "## Section\n\nbody") == "my notes"


def test_title_falls_back_to_a_readable_filename() -> None:
    assert title_for(Path("redshift_architecture.md"), "no heading") == "redshift architecture"


def test_title_collapses_whitespace_in_a_heading() -> None:
    assert title_for(Path("a.md"), "#   Spaced   Out  \n") == "Spaced Out"


def test_title_strips_closing_hashes() -> None:
    assert title_for(Path("a.md"), "# Closed Heading #\n") == "Closed Heading"


def test_type_comes_from_the_containing_folder(corpus: Path) -> None:
    assert type_for(corpus / "architecture" / "redshift.md", corpus) == "architecture"


def test_type_at_the_root_is_note_not_the_root_folder_name(corpus: Path) -> None:
    """The root is whatever folder the corpus lives in; it is not a document type."""
    assert type_for(corpus / "top.md", corpus) == "note"


def test_an_explicit_type_wins(corpus: Path) -> None:
    assert type_for(corpus / "architecture" / "redshift.md", corpus, "manual") == "manual"


# --- reading --------------------------------------------------------------


def test_read_document_describes_a_file(corpus: Path) -> None:
    document = read_document(corpus / "architecture" / "redshift.md", corpus, tags=("infra",))

    assert document.title == "Redshift"
    assert document.type == "architecture"
    assert document.source == "architecture/redshift.md"
    assert document.tags == ("infra",)
    assert "body" in document.content


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    """The endpoint answers 422; catching it here keeps a walk going."""
    blank = tmp_path / "blank.md"
    blank.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        read_document(blank, tmp_path)


def test_a_non_utf8_file_is_refused(tmp_path: Path) -> None:
    binary = tmp_path / "binary.md"
    binary.write_bytes(b"\xff\xfe\x00garbage")

    with pytest.raises(ValueError, match="UTF-8"):
        read_document(binary, tmp_path)


def test_read_documents_collects_failures_instead_of_raising(corpus: Path) -> None:
    """One bad file must not abandon the other thirty-nine."""
    (corpus / "blank.md").write_text("", encoding="utf-8")

    result = read_documents(discover(corpus), corpus)

    assert [skipped.path.name for skipped in result.skipped] == ["blank.md"]
    assert len(result.documents) == 3


def test_read_documents_applies_tags_and_type_to_every_document(corpus: Path) -> None:
    result = read_documents(discover(corpus), corpus, document_type="manual", tags=("bulk",))

    assert {document.type for document in result.documents} == {"manual"}
    assert all(document.tags == ("bulk",) for document in result.documents)
