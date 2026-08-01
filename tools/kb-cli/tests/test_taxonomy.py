"""The controlled vocabulary, which is the last word on anything a model says."""

import pytest

from kb_cli.taxonomy import (
    DEFAULT_TYPE,
    DOCUMENT_TYPES,
    MAX_TAGS,
    coerce_type,
    normalise_tags,
    normalise_type,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sop", "sop"),
        ("SOP", "sop"),
        ("  Architecture  ", "architecture"),
        ("incident report", "incident-report"),
        ("incident_report", "incident-report"),
        ("Incident-Report", "incident-report"),
    ],
)
def test_a_type_is_recognised_however_it_is_written(raw: str, expected: str) -> None:
    assert normalise_type(raw) == expected


@pytest.mark.parametrize("raw", ["runbook", "documentation", "", "   ", "note-ish"])
def test_a_type_outside_the_list_is_not_recognised(raw: str) -> None:
    """`None`, not a fallback — the caller needs to know a substitution happened."""
    assert normalise_type(raw) is None


def test_an_unrecognised_type_coerces_to_note() -> None:
    assert coerce_type("runbook") == DEFAULT_TYPE
    assert coerce_type(None) == DEFAULT_TYPE


def test_every_listed_type_survives_a_round_trip() -> None:
    """The list and the slug rule must agree, or a valid answer becomes `note`."""
    assert all(normalise_type(document_type) == document_type for document_type in DOCUMENT_TYPES)


def test_tags_are_slugged() -> None:
    assert normalise_tags(["Ansible", "Wazuh SIEM", "bare_metal"]) == (
        "ansible",
        "wazuh-siem",
        "bare-metal",
    )


def test_tags_are_deduplicated_in_first_seen_order() -> None:
    assert normalise_tags(["security", "Security", "wazuh", "SECURITY"]) == ("security", "wazuh")


def test_tags_drop_characters_that_are_not_allowed() -> None:
    assert normalise_tags(["c++", "node.js", ".NET", "!!!"]) == ("c", "nodejs", "net")


def test_the_cap_keeps_the_earliest_tags() -> None:
    """A model lists its most confident tag first, so the cap must bite at the end."""
    many = [f"tag{index}" for index in range(MAX_TAGS + 4)]
    assert normalise_tags(many) == tuple(many[:MAX_TAGS])


def test_the_cap_is_adjustable() -> None:
    assert normalise_tags(["a", "b", "c"], limit=2) == ("a", "b")


def test_empty_tags_disappear_rather_than_becoming_blanks() -> None:
    assert normalise_tags(["", "  ", "-", "ansible"]) == ("ansible",)
