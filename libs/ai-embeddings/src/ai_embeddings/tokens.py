"""Token counting and pre-flight batching.

Counting is `tiktoken` rather than a characters-over-four heuristic because the
numbers are used for two things that a heuristic gets wrong in opposite
directions: deciding a batch fits inside the provider's per-request budget, and
reporting cost into the audit trail. For OpenAI's models this is the same
encoder the API bills with, so the figures agree exactly.

Batching implements Design §3.1 step 5 — at most 96 inputs or 100 000 tokens per
request, whichever binds first. Batches stay contiguous and in order, so a caller
can zip the returned vectors straight back onto its chunks.

`tiktoken` fetches its encoder file on first use and caches it on disk. In an
air-gapped environment set `TIKTOKEN_CACHE_DIR` to a pre-populated directory.
"""

from collections.abc import Iterable, Sequence
from functools import lru_cache

import tiktoken

__all__ = ["batch_texts", "count_tokens", "encoding_for", "total_tokens"]


@lru_cache(maxsize=8)
def encoding_for(name: str) -> tiktoken.Encoding:
    """A cached encoder. Loading one costs a file read and some MB of tables."""
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    return len(encoding_for(encoding).encode(text))


def batch_texts(
    texts: Iterable[str],
    *,
    max_inputs: int,
    max_tokens: int,
    encoding: str = "cl100k_base",
) -> list[list[str]]:
    """Split `texts` into request-sized batches, preserving order.

    A single text over `max_tokens` still gets its own batch rather than being
    silently dropped; the provider port rejects it with a message naming the
    input, which is more use than a batch that is quietly one short.
    """
    if max_inputs < 1:
        raise ValueError("max_inputs must be at least 1")

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for text in texts:
        tokens = count_tokens(text, encoding)
        exceeds_budget = current and (
            len(current) >= max_inputs or current_tokens + tokens > max_tokens
        )
        if exceeds_budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens

    if current:
        batches.append(current)
    return batches


def total_tokens(texts: Sequence[str], encoding: str = "cl100k_base") -> int:
    return sum(count_tokens(text, encoding) for text in texts)
