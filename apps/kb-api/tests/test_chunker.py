"""The Phase 5 exit criteria, plus the behaviour they rest on.

Byte-identical output across runs and processes, nothing over the token
ceiling, and chunks that reconstruct their source once the overlap is removed —
checked against the real design documents rather than only synthetic input,
because the awkward cases (fenced ASCII trees, tables, headings with no body)
all came from those files.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from ai_embeddings.tokens import count_tokens
from kb_api.chunking import (
    ChunkerConfig,
    chunk_document,
    chunk_id,
    normalise,
    text_hash,
    to_new_chunk,
)
from kb_api.chunking.config import DEFAULT_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[3]
# `ai-kb/` is gitignored. Locally these are the Phase 5 exit cases; in CI the
# glob is empty and collection must not die on `max([])`.
KB_DOCS = sorted(REPO_ROOT.glob("ai-kb/*.md"))
LARGEST_KB_DOC = max(KB_DOCS, key=lambda path: path.stat().st_size) if KB_DOCS else None

PROSE = normalise(
    "# Guide\n\n"
    + "\n\n".join(f"Paragraph {index}. " + "word " * 60 for index in range(20))
    + "\n\n## Appendix\n\n"
    + "\n\n".join(f"Appendix paragraph {index}. " + "term " * 60 for index in range(10))
)


def _two_sections(*, deep_words: int) -> str:
    """A document whose first section stands alone, so `## Deep` gets its own
    group rather than being merged into it."""
    return normalise(f"# Guide\n\n{'intro ' * 300}\n\n## Deep\n\n{'word ' * deep_words}")


def _document_ids() -> list[str]:
    return [path.name for path in KB_DOCS] or ["missing-ai-kb"]


@pytest.fixture(params=KB_DOCS or [None], ids=_document_ids())
def kb_document(request: pytest.FixtureRequest) -> str:
    path: Path | None = request.param
    if path is None:
        pytest.skip("ai-kb/*.md is not in this checkout")
    return path.read_text()


def test_the_design_documents_produce_chunks(kb_document):
    assert chunk_document(kb_document)


def test_chunks_reconstruct_the_source(kb_document):
    drafts = chunk_document(kb_document)
    assert "".join(draft.source for draft in drafts) == normalise(kb_document)


def test_no_chunk_exceeds_the_token_ceiling(kb_document):
    for draft in chunk_document(kb_document):
        assert draft.token_count <= DEFAULT_CONFIG.max_tokens


def test_no_chunk_falls_below_the_minimum(kb_document):
    """Design §4: below `min_tokens` a chunk merges into its neighbour.

    A document that chunks to a single piece is exempt — there is no neighbour.
    """
    drafts = chunk_document(kb_document)
    if len(drafts) > 1:
        assert min(draft.token_count for draft in drafts) >= DEFAULT_CONFIG.min_tokens


def test_ordinals_are_dense_and_ordered(kb_document):
    drafts = chunk_document(kb_document)
    assert [draft.ordinal for draft in drafts] == list(range(len(drafts)))


def test_output_is_identical_across_repeated_runs(kb_document):
    assert chunk_document(kb_document) == chunk_document(kb_document)


def test_output_is_identical_across_processes(tmp_path: Path):
    """`PYTHONHASHSEED` is what makes this worth a subprocess.

    A chunker that iterated a set or a dict keyed by text would be stable
    within a process and unstable across restarts, which AD-008 turns into a
    full re-embed of the corpus on every deploy.
    """
    source = LARGEST_KB_DOC if LARGEST_KB_DOC is not None else tmp_path / "prose.md"
    if LARGEST_KB_DOC is None:
        source.write_text(PROSE)

    script = (
        "import hashlib,sys;"
        "from pathlib import Path;"
        "from kb_api.chunking import chunk_document;"
        "drafts=chunk_document(Path(sys.argv[1]).read_text());"
        "print(hashlib.sha256('\\x00'.join(d.text for d in drafts).encode()).hexdigest())"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script, str(source)],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(digests) == 1

    local = hashlib.sha256(
        "\x00".join(draft.text for draft in chunk_document(source.read_text())).encode()
    ).hexdigest()
    assert digests == {local}


def test_editing_one_paragraph_leaves_the_other_chunks_alone():
    """The AD-008 payoff, at the chunker level.

    Greedy packing is what makes this true: a balanced splitter would move
    every boundary after the edit and re-embed the rest of the document.
    """
    before = chunk_document(PROSE)
    edited = PROSE.replace("Paragraph 3.", "Paragraph 3, revised.")
    after = chunk_document(edited)

    unchanged = {draft.text for draft in before} & {draft.text for draft in after}
    assert len(unchanged) >= len(before) - 2


def test_a_chunk_carries_its_heading_path():
    drafts = chunk_document(PROSE)
    assert drafts[-1].heading_path == ("Guide", "Appendix")
    assert drafts[-1].text.startswith("# Guide > ## Appendix\n\n")


def test_the_owning_heading_is_not_repeated_in_the_breadcrumb():
    """The first chunk of a section already opens with that heading line."""
    section = chunk_document(_two_sections(deep_words=400))[1]
    assert section.text.startswith("# Guide\n\n## Deep\n")
    assert section.text.count("Deep") == 1


def test_later_pieces_of_a_split_section_keep_the_full_breadcrumb():
    drafts = chunk_document(_two_sections(deep_words=900))
    assert drafts[2].text.startswith("# Guide > ## Deep\n\n")


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    drafts = chunk_document(PROSE)
    overlapping = [draft for draft in drafts if draft.overlap_chars]
    assert overlapping
    for draft in overlapping:
        previous = drafts[draft.ordinal - 1]
        overlap = draft.body[: draft.overlap_chars]
        assert previous.source.endswith(overlap)
        assert count_tokens(overlap) <= DEFAULT_CONFIG.overlap_tokens


def test_a_block_with_no_whitespace_still_splits():
    """The last-resort token split — no paragraph or sentence boundary exists."""
    drafts = chunk_document("x" * 40_000)
    assert len(drafts) > 1
    assert "".join(draft.source for draft in drafts) == normalise("x" * 40_000)
    for draft in drafts:
        assert draft.token_count <= DEFAULT_CONFIG.max_tokens


def test_multibyte_text_survives_a_token_split():
    source = "日本語テキスト" * 2_000
    drafts = chunk_document(source)
    assert len(drafts) > 1
    assert "".join(draft.source for draft in drafts) == normalise(source)
    assert "�" not in "".join(draft.body for draft in drafts)


def test_an_empty_document_chunks_to_nothing():
    assert chunk_document("") == ()
    assert chunk_document("   \n\n") == ()


def test_a_tighter_config_produces_more_chunks():
    tight = ChunkerConfig(target_tokens=128, overlap_tokens=16, min_tokens=32)
    assert len(chunk_document(PROSE, tight)) > len(chunk_document(PROSE))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 0},
        {"overlap_tokens": 512},
        {"min_tokens": 600},
        {"max_tokens": 100},
    ],
)
def test_an_incoherent_config_is_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        ChunkerConfig(**kwargs)


def test_to_new_chunk_derives_identity_from_content():
    document_id = uuid4()
    draft = chunk_document(PROSE)[0]
    new_chunk = to_new_chunk(draft, document_id=document_id, model_id="text-embedding-3-small")

    assert new_chunk.text_hash == text_hash(draft.text, "text-embedding-3-small")
    assert new_chunk.id == chunk_id(document_id, draft.ordinal, new_chunk.text_hash)
    assert new_chunk.chroma_id == str(new_chunk.id)
    assert new_chunk.token_count == draft.token_count


def test_the_same_draft_and_document_always_bind_to_the_same_id():
    document_id = uuid4()
    draft = chunk_document(PROSE)[0]
    first = to_new_chunk(draft, document_id=document_id, model_id="m")
    second = to_new_chunk(draft, document_id=document_id, model_id="m")
    assert first == second


def test_a_different_model_gives_a_different_hash_and_id():
    document_id = uuid4()
    draft = chunk_document(PROSE)[0]
    openai = to_new_chunk(draft, document_id=document_id, model_id="text-embedding-3-small")
    gemini = to_new_chunk(draft, document_id=document_id, model_id="gemini-embedding-001")
    assert openai.text_hash != gemini.text_hash
    assert openai.id != gemini.id


def test_the_hash_separator_keeps_text_and_model_apart():
    assert text_hash("ab", "cd") != text_hash("a", "bcd")
