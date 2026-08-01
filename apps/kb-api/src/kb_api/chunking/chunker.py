"""The chunker: normalised Markdown in, embeddable chunks out.

Three passes. Split on headings; group adjacent sections until the group is
worth embedding on its own; split any group that is too large, with overlap
between the pieces. Nothing consults a clock, a random source, or a hash-ordered
container, because AD-008 makes chunk boundaries part of the cache key — a
chunker that returned different output on a second run would re-embed the whole
corpus every time it ran.

Each draft carries both what gets embedded (`text`, heading breadcrumb
included) and the verbatim source span it came from (`body`), with
`overlap_chars` marking how much of the body is repeated from the previous
chunk. Dropping that prefix from every body and concatenating reproduces the
input exactly, which is how the exit criteria check that nothing was lost.
"""

from dataclasses import dataclass
from uuid import UUID

from ai_embeddings.tokens import count_tokens
from kb_api.chunking.config import DEFAULT_CONFIG, ChunkerConfig
from kb_api.chunking.hashing import chroma_id, chunk_id, text_hash
from kb_api.chunking.markdown import Section, common_path, render_prefix, split_sections
from kb_api.chunking.normalise import normalise
from kb_api.chunking.splitter import split_to_budget, token_suffix
from kb_api.domain import NewChunk

__all__ = ["ChunkDraft", "chunk_document", "to_new_chunk"]

_PREFIX_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """One chunk, before it has a document to belong to."""

    ordinal: int
    text: str
    body: str
    heading_path: tuple[str, ...]
    token_count: int
    overlap_chars: int

    @property
    def source(self) -> str:
        """The part of `body` that is this chunk's own, not the overlap tail."""
        return self.body[self.overlap_chars :]


def chunk_document(content: str, config: ChunkerConfig = DEFAULT_CONFIG) -> tuple[ChunkDraft, ...]:
    """Chunk a document. `content` is normalised again here — `normalise` is
    idempotent, and a caller that forgot would otherwise get boundaries that
    depend on its line endings."""
    text = normalise(content)
    if not text:
        return ()

    drafts: list[ChunkDraft] = []
    for group in _group_sections(split_sections(text), config):
        _extend(drafts, group, config)
    return tuple(drafts)


def to_new_chunk(draft: ChunkDraft, *, document_id: UUID, model_id: str) -> NewChunk:
    """Bind a draft to a document and a model, deriving its identity (AD-008)."""
    hashed = text_hash(draft.text, model_id)
    identifier = chunk_id(document_id, draft.ordinal, hashed)
    return NewChunk(
        id=identifier,
        document_id=document_id,
        ordinal=draft.ordinal,
        text=draft.text,
        text_hash=hashed,
        token_count=draft.token_count,
        chroma_id=chroma_id(identifier),
    )


def _group_sections(
    sections: tuple[Section, ...], config: ChunkerConfig
) -> list[tuple[Section, ...]]:
    """Merge adjacent sections until a group is at least `min_tokens`.

    This is Design §4's "below which a chunk merges into its neighbour". A
    heading with two lines under it is not worth a vector of its own, and the
    neighbour it merges with is the next section rather than an arbitrary one,
    so the merged text stays contiguous in the source. The merged chunk is
    labelled with the deepest heading path all its sections share, which is the
    only breadcrumb that is true of all of it.

    A group is allowed past `target_tokens` on the way to `min_tokens`: a bare
    `## Data Model` heading followed by a section of its own that is already
    oversized has nowhere else to go, and the splitter below re-cuts the
    combined text anyway. Stopping short would leave the heading as a chunk of
    nothing but itself.
    """
    groups: list[tuple[Section, ...]] = []
    current: list[Section] = []
    current_tokens = 0

    for section in sections:
        if current and current_tokens >= config.min_tokens:
            groups.append(tuple(current))
            current, current_tokens = [], 0
        current.append(section)
        current_tokens += count_tokens(section.body, config.encoding)

    if current:
        groups.append(tuple(current))
    # The loop flushes on the group *before* it knows the group is the last
    # one, so a short tail section arrives here as its own group. It merges
    # backwards when there is room; when there is not, a slightly short chunk
    # beats one over target.
    if len(groups) > 1 and current_tokens < config.min_tokens:
        merged = groups[-2] + groups[-1]
        if count_tokens("".join(s.body for s in merged), config.encoding) <= config.target_tokens:
            groups[-2:] = [merged]
    return groups


def _merge_runts(spans: list[str], config: ChunkerConfig) -> list[str]:
    """Fold pieces below `min_tokens` into an adjacent piece.

    Two sources of runts. Greedy packing leaves the remainder in the last
    piece, which for a section a few tokens past budget is one sentence. And a
    heading line followed immediately by an oversized block — a long fenced
    diagram — flushes the heading on its own before the block is split.

    Merging costs a chunk slightly over target; the alternative is an embedding
    call spent on a fragment with no content, which is worse in results and
    costs the same.
    """
    merged: list[str] = []
    for span in spans:
        if merged and count_tokens(span, config.encoding) < config.min_tokens:
            merged[-1] += span
        else:
            merged.append(span)
    if len(merged) > 1 and count_tokens(merged[0], config.encoding) < config.min_tokens:
        merged[:2] = [merged[0] + merged[1]]
    return merged


def _extend(drafts: list[ChunkDraft], group: tuple[Section, ...], config: ChunkerConfig) -> None:
    heading_path = common_path(tuple(section.heading_path for section in group))
    body = "".join(section.body for section in group)
    # The group's own heading line opens its body, so repeating it as the last
    # breadcrumb element would embed the same words twice. Ancestors are not in
    # the body and stay. Later pieces of a split group start mid-body and so
    # keep the full path.
    lead = heading_path[:-1] if heading_path == group[0].heading_path else heading_path
    prefixes = (render_prefix(lead), render_prefix(heading_path))

    prefix_tokens = count_tokens(prefixes[1] + _PREFIX_SEPARATOR, config.encoding)
    # The breadcrumb and the overlap tail both eat into the target, and a deep
    # enough heading path could otherwise leave no room for content at all.
    budget = max(config.target_tokens - prefix_tokens - config.overlap_tokens, config.min_tokens, 1)

    previous = ""
    for index, span in enumerate(
        _merge_runts(split_to_budget(body, budget=budget, encoding=config.encoding), config)
    ):
        overlap = (
            token_suffix(previous, max_tokens=config.overlap_tokens, encoding=config.encoding)
            if previous
            else ""
        )
        prefix = prefixes[min(index, 1)]
        chunk_body = overlap + span
        chunk_text = f"{prefix}{_PREFIX_SEPARATOR}{chunk_body}" if prefix else chunk_body
        drafts.append(
            ChunkDraft(
                ordinal=len(drafts),
                text=chunk_text,
                body=chunk_body,
                heading_path=heading_path,
                token_count=count_tokens(chunk_text, config.encoding),
                overlap_chars=len(overlap),
            )
        )
        previous = span
