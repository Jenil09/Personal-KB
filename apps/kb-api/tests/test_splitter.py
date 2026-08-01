"""Token-bounded splitting, level by level.

`chunk_document` exercises these through real documents; these cases pin the
individual levels, where a regression is otherwise visible only as a slightly
different chunk count.
"""

import pytest

from ai_embeddings.tokens import count_tokens
from kb_api.chunking.splitter import split_to_budget, token_suffix

PARAGRAPHS = "\n\n".join(f"Paragraph {index}. " + "word " * 40 for index in range(10))


def _pieces(text: str, budget: int) -> list[str]:
    pieces = split_to_budget(text, budget=budget, encoding="cl100k_base")
    assert "".join(pieces) == text
    return pieces


def test_text_within_budget_is_left_whole():
    assert _pieces(PARAGRAPHS, 10_000) == [PARAGRAPHS]


def test_prose_splits_on_paragraph_boundaries():
    for piece in _pieces(PARAGRAPHS, 200):
        assert piece.lstrip().startswith("Paragraph")


def test_a_single_oversized_paragraph_splits_on_sentences():
    paragraph = " ".join(f"Sentence number {index} says something." for index in range(200))
    pieces = _pieces(paragraph, 100)
    assert len(pieces) > 1
    for piece in pieces[:-1]:
        assert piece.rstrip().endswith(".")


def test_a_sentence_with_no_boundary_splits_on_tokens():
    pieces = _pieces("x" * 8_000, 100)
    assert len(pieces) > 1
    assert all(count_tokens(piece) <= 100 for piece in pieces)


def test_multibyte_characters_are_never_cut_in_half():
    source = "日本語" * 3_000
    pieces = _pieces(source, 50)
    assert "�" not in "".join(pieces)


def test_an_empty_string_is_one_empty_piece():
    assert split_to_budget("", budget=100, encoding="cl100k_base") == [""]


def test_a_budget_below_one_is_rejected():
    with pytest.raises(ValueError):
        split_to_budget("text", budget=0, encoding="cl100k_base")


def test_a_suffix_shorter_than_the_budget_is_the_whole_text():
    assert token_suffix("short text", max_tokens=64, encoding="cl100k_base") == "short text"


def test_a_suffix_is_a_real_suffix_within_its_budget():
    suffix = token_suffix(PARAGRAPHS, max_tokens=32, encoding="cl100k_base")
    assert PARAGRAPHS.endswith(suffix)
    assert count_tokens(suffix) <= 32


def test_a_suffix_starts_at_a_word_boundary():
    suffix = token_suffix("alpha beta gamma delta " * 100, max_tokens=8, encoding="cl100k_base")
    assert suffix.split()[0] in {"alpha", "beta", "gamma", "delta"}


def test_a_multibyte_suffix_decodes_cleanly():
    suffix = token_suffix("日本語テキスト" * 500, max_tokens=16, encoding="cl100k_base")
    assert "�" not in suffix
    assert ("日本語テキスト" * 500).endswith(suffix)


def test_no_suffix_is_asked_for_when_the_budget_is_zero():
    assert token_suffix("text", max_tokens=0, encoding="cl100k_base") == ""
    assert token_suffix("", max_tokens=64, encoding="cl100k_base") == ""
