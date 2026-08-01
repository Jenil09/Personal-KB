"""Chunker configuration and its version.

Every value here is baked into the collection name (Design §2.3), because
changing any of them moves chunk boundaries, which invalidates every
`text_hash` and therefore every vector in the store (AD-008). Bumping
`CHUNKER_VERSION` is the deliberate act that says "this is a different index";
changing a default without bumping it silently corrupts carry-forward reuse,
since old hashes would still match text the chunker would no longer produce.
"""

from dataclasses import dataclass

__all__ = ["CHUNKER_VERSION", "ChunkerConfig"]

# Increment when any chunking behaviour changes — defaults, splitting rules, or
# the heading-path prefix format.
CHUNKER_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Design §4 defaults.

    `target_tokens` is a target rather than a limit: a chunk carrying a long
    heading path or an overlap tail can land slightly over it. `max_tokens` is
    the limit, and it is the embedding model's input ceiling — exceeding it is
    an API error, so the splitter guarantees it by construction.
    """

    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 128
    max_tokens: int = 8191
    encoding: str = "cl100k_base"

    def __post_init__(self) -> None:
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be at least 1")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be non-negative and below target_tokens")
        if not 0 <= self.min_tokens <= self.target_tokens:
            raise ValueError("min_tokens must be between zero and target_tokens")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be at least target_tokens")


DEFAULT_CONFIG = ChunkerConfig()
