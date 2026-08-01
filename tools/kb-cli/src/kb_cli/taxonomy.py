"""The controlled vocabulary the suggester answers in.

**Client-side only, deliberately.** `kb-api` accepts any `type` string and
`ingest.type_for()` still infers one from the directory a file sits in, so a
`runbooks/` folder keeps producing `runbook` on an `ingest-dir` walk. Nothing
here changes that. This list exists so an LLM asked to classify a document
answers from a closed set rather than inventing a taxonomy per document — which
is the whole difference between suggestions that accumulate into a usable filter
and suggestions that produce forty singleton types.

The consequence to keep in mind: the corpus can contain types that are not in
this list, and a suggestion for such a document will move it into the list. That
is the intended direction of travel, not a bug.
"""

import re
from collections.abc import Iterable

__all__ = ["DEFAULT_TYPE", "DOCUMENT_TYPES", "coerce_type", "normalise_tags", "normalise_type"]

DOCUMENT_TYPES: tuple[str, ...] = (
    "profile",
    "architecture",
    "incident-report",
    "sop",
    "philosophy",
    "migration",
    "blog",
    "note",
)

DEFAULT_TYPE = "note"
"""Where an unrecognised suggestion lands.

The same fallback `ingest.type_for()` uses for a file with no directory signal,
so a document that nothing could classify is filed the same way whichever path
it arrived by.
"""

MAX_TAGS = 6
"""More than this stops being metadata and starts being a summary.

Six is generous for a corpus of tens of documents: a tag that appears on one
document filters nothing, and a model given no ceiling will happily emit twelve.
"""

MAX_TAG_CHARS = 32

_SEPARATORS = re.compile(r"[\s_/]+")
_DISALLOWED = re.compile(r"[^a-z0-9-]+")
_RUNS = re.compile(r"-{2,}")


def _slug(value: str) -> str:
    """Lowercase, hyphen-separated, `[a-z0-9-]` only.

    Applied to both types and tags so `Incident Report`, `incident_report`, and
    `INCIDENT-REPORT` are one answer rather than three.
    """
    lowered = _SEPARATORS.sub("-", value.strip().lower())
    return _RUNS.sub("-", _DISALLOWED.sub("", lowered)).strip("-")


def normalise_type(raw: str | None) -> str | None:
    """The canonical type `raw` names, or `None` if it names none.

    `None` rather than a fallback because the caller needs to distinguish the
    two: the metadata screen says "coerced from `runbook`" only when it knows a
    substitution happened.
    """
    if not raw:
        return None
    slug = _slug(raw)
    return slug if slug in DOCUMENT_TYPES else None


def coerce_type(raw: str | None) -> str:
    """`normalise_type`, with `note` for anything unrecognised."""
    return normalise_type(raw) or DEFAULT_TYPE


def normalise_tags(values: Iterable[str], *, limit: int = MAX_TAGS) -> tuple[str, ...]:
    """Slugged, de-duplicated, capped — order preserved.

    Order is preserved rather than sorted because a model lists its most
    confident tag first, and that is the one to keep when the cap bites.
    """
    seen: dict[str, None] = {}
    for value in values:
        slug = _slug(value)[:MAX_TAG_CHARS].strip("-")
        if slug:
            seen.setdefault(slug, None)
    return tuple(seen)[:limit]
