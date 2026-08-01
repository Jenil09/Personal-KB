"""Stage one: structural split on Markdown headings (Design §4).

A section's `body` is the verbatim source span from its heading line up to the
next heading, so the bodies of all sections concatenate back to the input
exactly. That is what makes the reconstruction property testable and what lets
adjacent sections merge by string concatenation later.

The heading path is carried separately because it is prefixed onto the embedded
text rather than found in it — a chunk taken from deep inside a document
otherwise arrives at the model with no idea what it is about.

Fenced code blocks are tracked because `# comment` at the start of a shell line
inside a fence is not a heading, and treating it as one would split a code block
in half.
"""

import re
from dataclasses import dataclass

__all__ = ["Section", "common_path", "render_prefix", "split_sections"]

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class Section:
    """One heading and everything under it, down to the next heading.

    `heading_path` is the enclosing headings outermost-first, including this
    section's own. It is empty for content appearing before the first heading.
    """

    heading_path: tuple[str, ...]
    body: str


def split_sections(content: str) -> tuple[Section, ...]:
    """Split normalised Markdown into heading-delimited sections.

    Content before the first heading becomes a section with an empty path
    rather than being attached to the first heading, which would give a
    document's preamble a breadcrumb it does not belong under.
    """
    if not content:
        return ()

    sections: list[Section] = []
    path: list[str] = []
    current: list[str] = []
    current_path: tuple[str, ...] = ()
    fence: str | None = None

    def flush() -> None:
        if current:
            sections.append(Section(current_path, "".join(current)))
            current.clear()

    for line in content.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        opener = _FENCE.match(stripped)
        if fence is not None:
            if opener and opener.group(1).startswith(fence):
                fence = None
            current.append(line)
            continue
        if opener:
            fence = opener.group(1)
            current.append(line)
            continue

        heading = _HEADING.match(stripped)
        if heading is None:
            current.append(line)
            continue

        flush()
        level = len(heading.group(1))
        # A jump from `#` straight to `###` keeps the path at the depth the
        # document actually has rather than inventing empty ancestors.
        del path[level - 1 :]
        path.append(heading.group(2))
        current_path = tuple(path)
        current.append(line)

    flush()
    return tuple(sections)


def common_path(paths: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """The longest heading path every one of `paths` starts with.

    Used when adjacent sections merge into one chunk: the merged chunk can only
    honestly claim the headings all of its content sits under.
    """
    if not paths:
        return ()
    shared: list[str] = []
    for headings in zip(*paths, strict=False):
        first = headings[0]
        if any(heading != first for heading in headings):
            break
        shared.append(first)
    return tuple(shared)


def render_prefix(heading_path: tuple[str, ...]) -> str:
    """The breadcrumb prepended to a chunk's text, e.g. `# A > ## B > ### C`.

    Part of the chunker's versioned behaviour: changing this format changes
    every embedded string and so requires a `CHUNKER_VERSION` bump.
    """
    return " > ".join(
        f"{'#' * (depth + 1)} {heading}" for depth, heading in enumerate(heading_path)
    )
