"""Normalisation decides what counts as the same document (AD-008)."""

from kb_api.chunking import content_hash, normalise


def test_line_endings_collapse_to_lf():
    assert normalise("a\r\nb\rc\n") == "a\nb\nc\n"


def test_the_same_document_from_two_editors_hashes_the_same():
    windows = "# Title\r\n\r\nA line with trailing space   \r\n"
    unix = "# Title\n\nA line with trailing space\n\n\n"
    assert content_hash(normalise(windows)) == content_hash(normalise(unix))


def test_indentation_and_blank_lines_inside_the_document_survive():
    source = "para one\n\npara two\n\n```\n    indented\n```\n"
    assert normalise(source) == source


def test_a_whitespace_only_document_normalises_to_empty():
    assert normalise("  \n\n\t\n") == ""


def test_normalise_is_idempotent():
    once = normalise("# T\r\n\r\ntext   \n\n")
    assert normalise(once) == once


def test_a_byte_order_mark_is_not_part_of_the_content():
    assert normalise("﻿# Title\n") == "# Title\n"
