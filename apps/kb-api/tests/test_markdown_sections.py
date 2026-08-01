"""Structural split: heading paths, and what is not a heading."""

from kb_api.chunking.markdown import common_path, render_prefix, split_sections
from kb_api.chunking.normalise import normalise

NESTED = normalise(
    """
preamble text

# Redshift Architecture

intro

## Networking

net text

### Cilium

cilium text

## Storage

storage text
"""
)


def test_sections_partition_the_source_exactly():
    assert "".join(section.body for section in split_sections(NESTED)) == NESTED


def test_heading_paths_nest_and_unwind():
    paths = [section.heading_path for section in split_sections(NESTED)]
    assert paths == [
        (),
        ("Redshift Architecture",),
        ("Redshift Architecture", "Networking"),
        ("Redshift Architecture", "Networking", "Cilium"),
        ("Redshift Architecture", "Storage"),
    ]


def test_content_before_the_first_heading_gets_an_empty_path():
    assert split_sections(NESTED)[0].heading_path == ()


def test_a_skipped_level_does_not_invent_an_ancestor():
    sections = split_sections(normalise("# One\n\n### Three\n\ntext\n"))
    assert sections[1].heading_path == ("One", "Three")


def test_a_comment_inside_a_fence_is_not_a_heading():
    source = normalise("# Real\n\n```bash\n# not a heading\necho hi\n```\n\nafter\n")
    sections = split_sections(source)
    assert len(sections) == 1
    assert "# not a heading" in sections[0].body


def test_a_tilde_fence_containing_a_backtick_fence_closes_only_on_tildes():
    source = normalise("# Real\n\n~~~\n```\n# still code\n```\n~~~\n\n# Next\n\nx\n")
    assert [section.heading_path for section in split_sections(source)] == [("Real",), ("Next",)]


def test_closing_hashes_are_not_part_of_the_heading():
    assert split_sections(normalise("## Title ##\n\nx\n"))[0].heading_path == ("Title",)


def test_a_hash_without_a_space_is_not_a_heading():
    assert split_sections(normalise("#hashtag\n"))[0].heading_path == ()


def test_empty_content_has_no_sections():
    assert split_sections("") == ()


def test_common_path_is_the_shared_prefix():
    assert common_path((("A", "B", "C"), ("A", "B"), ("A", "B", "D"))) == ("A", "B")
    assert common_path((("A",), ("B",))) == ()
    assert common_path(()) == ()


def test_the_prefix_renders_as_a_breadcrumb():
    assert render_prefix(("A", "B", "C")) == "# A > ## B > ### C"
    assert render_prefix(()) == ""
