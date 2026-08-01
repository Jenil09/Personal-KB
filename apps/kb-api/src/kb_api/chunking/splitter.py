"""Stage two: token-bounded splitting (Design §4).

Descends through progressively worse split points — paragraphs, then sentences,
then raw token boundaries — and only goes a level deeper when the level above
produced a piece that still does not fit. Prose therefore splits between
paragraphs almost always, and a minified JSON blob with no whitespace in it
still splits rather than being handed to the model over its input limit.

Every function here returns pieces that concatenate back to their input
character for character. The reconstruction property in the Phase 5 exit
criteria depends on it, and so does the overlap accounting in `chunker.py`,
which records overlap as a character count.
"""

import codecs
import re
from collections.abc import Callable, Iterable

from ai_embeddings.tokens import count_tokens, encoding_for

__all__ = ["split_to_budget", "token_suffix"]

# Each piece keeps its trailing separator, so joining is exact.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
# The curly quotes are deliberate: prose from an editor closes a quoted
# sentence with them, and a split point before the closer strands it.
_SENTENCE_BREAK = re.compile(r"""(?<=[.!?…])["'”’)\]]*\s+""")  # noqa: RUF001

_Splitter = Callable[[str, int, str], list[str]]


def split_to_budget(text: str, *, budget: int, encoding: str) -> list[str]:
    """Split `text` into pieces of at most roughly `budget` tokens each."""
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if not text or count_tokens(text, encoding) <= budget:
        return [text]
    return _pack(_split_keep(text, _PARAGRAPH_BREAK), budget, encoding, _split_paragraph)


def token_suffix(text: str, *, max_tokens: int, encoding: str) -> str:
    """The trailing `max_tokens` tokens of `text`, as an exact suffix string.

    Used for chunk overlap. Trimmed forward to the first whitespace so the
    overlap starts at a word boundary — a chunk beginning mid-word reads as
    corruption to anyone inspecting search results, and costs a token for
    nothing.
    """
    if max_tokens < 1 or not text:
        return ""
    enc = encoding_for(encoding)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    raw = enc.decode_bytes(tokens[-max_tokens:])
    # A token boundary can land mid-character; drop the leading continuation
    # bytes rather than decoding them to a replacement char, which would stop
    # the overlap being a real suffix of the source.
    start = 0
    while start < len(raw) and raw[start] & 0xC0 == 0x80:
        start += 1
    suffix = raw[start:].decode()

    match = re.search(r"\s", suffix)
    return suffix[match.end() :] if match else suffix


def _split_paragraph(paragraph: str, budget: int, encoding: str) -> list[str]:
    """Only reached for a paragraph already known to be over budget."""
    return _pack(_split_keep(paragraph, _SENTENCE_BREAK), budget, encoding, _split_by_tokens)


def _split_by_tokens(text: str, budget: int, encoding: str) -> list[str]:
    """The last resort: cut on token boundaries.

    Decoding is incremental because a character can span two tokens, and a
    piece that ends mid-character must carry those bytes into the next piece
    instead of emitting U+FFFD.
    """
    enc = encoding_for(encoding)
    tokens = enc.encode(text)
    decoder = codecs.getincrementaldecoder("utf-8")()
    pieces = [
        decoder.decode(enc.decode_bytes(tokens[start : start + budget]))
        for start in range(0, len(tokens), budget)
    ]
    return [piece for piece in pieces if piece]


def _pack(units: Iterable[str], budget: int, encoding: str, split_further: _Splitter) -> list[str]:
    """Greedily fill pieces with whole units, in order.

    Greedy rather than balanced: a balanced packer would move a boundary in the
    middle of a document when text is appended at the end, re-chunking and
    re-embedding everything after the edit for no gain (AD-008).
    """
    pieces: list[str] = []
    current = ""
    current_tokens = 0

    for unit in units:
        tokens = count_tokens(unit, encoding)
        if tokens > budget:
            if current:
                pieces.append(current)
                current, current_tokens = "", 0
            pieces.extend(split_further(unit, budget, encoding))
            continue
        if current and current_tokens + tokens > budget:
            pieces.append(current)
            current, current_tokens = "", 0
        current += unit
        current_tokens += tokens

    if current:
        pieces.append(current)
    return pieces


def _split_keep(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on `pattern`, keeping each separator on the piece it follows."""
    pieces: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        pieces.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces
