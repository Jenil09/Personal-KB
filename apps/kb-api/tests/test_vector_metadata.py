"""Chunk metadata, which is scalars only (AD-005).

Chroma rejects a list metadata value outright, so the one interesting property
here is that nothing this function builds is a list — including `tags`, which is
the field that most wants to be one and is instead a pipe-delimited display
string that is never filtered on.
"""

from uuid import UUID

import pytest

from kb_api.domain import DOCUMENT_ID_KEY, chunk_metadata, parse_tags, render_tags

DOCUMENT_ID = UUID("3f2504e0-4f89-41d3-9a0c-0305e82c3301")


def build(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "document_id": DOCUMENT_ID,
        "title": "Redshift Architecture",
        "source": "redshift.md",
        "document_type": "architecture",
        "tags": ("ansible", "hardening"),
        "ordinal": 3,
    }
    values.update(overrides)
    return dict(chunk_metadata(**values))  # type: ignore[arg-type]


def test_every_value_is_a_scalar_chroma_accepts() -> None:
    # The whole of AD-005 in one assertion: a list here is a rejected upsert,
    # and the rejection happens after the tokens have been spent.
    assert all(isinstance(value, str | int | float | bool) for value in build().values())


def test_the_document_id_is_a_string_because_the_filter_compares_strings() -> None:
    # `delete_document` and the tag `$in` clause both match on this value, and
    # Chroma compares a UUID object to a stored string as unequal.
    assert build()[DOCUMENT_ID_KEY] == str(DOCUMENT_ID)


def test_tags_are_pipe_delimited_for_display() -> None:
    assert build()["tags"] == "ansible|hardening"


def test_a_document_without_a_source_carries_no_source_key() -> None:
    # Not `"None"`, and not a null. A `where` clause on `source` should fail to
    # match this document rather than match the string "None".
    assert "source" not in build(source=None)


def test_the_ordinal_stays_an_integer() -> None:
    assert build()["ordinal"] == 3


@pytest.mark.parametrize("tags", [(), ("one",), ("a", "b", "c")])
def test_rendering_tags_round_trips(tags: tuple[str, ...]) -> None:
    assert parse_tags(render_tags(tags)) == tags


def test_no_tags_parses_to_no_tags_rather_than_one_empty_one() -> None:
    assert parse_tags("") == ()
