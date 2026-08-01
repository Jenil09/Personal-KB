"""Content normalisation — step 1 of the ingestion flow (Design §3.1).

Everything downstream hashes normalised text, so this runs before
`content_hash` and before chunking. The point is that the same document
submitted from Windows and from Linux, or with an editor's trailing whitespace
added, is the same document and must not re-embed.

Deliberately conservative. Whitespace inside a line and blank lines between
paragraphs are load-bearing in Markdown — collapsing them would change what a
fenced code block contains, and a normaliser that edits content is worse than
one that misses a duplicate.
"""

__all__ = ["normalise"]

_BOM = "﻿"


def normalise(content: str) -> str:
    """CRLF/CR to LF, no trailing whitespace on any line, single trailing newline.

    An empty or whitespace-only document normalises to the empty string rather
    than to `"\\n"`, so "nothing" has exactly one representation and one hash.
    """
    text = content.removeprefix(_BOM).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    body = "\n".join(lines).strip("\n")
    return f"{body}\n" if body else ""
