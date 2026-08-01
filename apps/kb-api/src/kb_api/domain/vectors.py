"""What the service hands a vector store, and what it gets back.

Chroma's own types do not appear above `adapters/chroma`. That is the whole
point of AD-010 keeping the adapter behind a port: a `QueryResult` of parallel
lists — ids here, distances there, metadata somewhere else, any of them possibly
`None` — is a shape to translate at the boundary, not one to carry through the
service and into a router.

Metadata is scalars only, because Chroma rejects lists (AD-005). `tags` is
therefore pipe-delimited and **display only** — it is returned in a search
result and never filtered on. Tag filtering resolves through Postgres.
"""

from dataclasses import dataclass
from uuid import UUID

from ai_embeddings import EmbeddingVector

__all__ = [
    "DOCUMENT_ID_KEY",
    "TAG_SEPARATOR",
    "ChunkMetadata",
    "MatchMetadata",
    "VectorMatch",
    "VectorRecord",
    "chunk_metadata",
    "parse_tags",
    "read_metadata",
    "render_tags",
]

TAG_SEPARATOR = "|"

DOCUMENT_ID_KEY = "document_id"
"""The metadata key a document is purged and tag-filtered by (AD-005, §3.3)."""

# The scalar types Chroma accepts as metadata values.
MetadataValue = str | int | float | bool
ChunkMetadata = dict[str, MetadataValue]


def render_tags(tags: tuple[str, ...]) -> str:
    """Pipe-delimit tags for display (AD-005). Never a filter input."""
    return TAG_SEPARATOR.join(tags)


def parse_tags(rendered: str) -> tuple[str, ...]:
    """Undo `render_tags`. Empty string is no tags, not one empty tag."""
    return tuple(tag for tag in rendered.split(TAG_SEPARATOR) if tag)


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One chunk on its way into the index.

    `chunk_text` is stored alongside the vector because AD-004 builds the search
    response from the Chroma payload alone — a result that had to fetch its own
    text from Postgres would put the system of record back on the hot path.
    """

    id: str
    vector: EmbeddingVector
    chunk_text: str
    metadata: ChunkMetadata


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """One search hit, still carrying Chroma's distance rather than a score.

    The distance-to-score mapping is the search service's (Design §3.2 step 6),
    not the store's: a store that returned similarity would be deciding which
    space it is in on the caller's behalf.
    """

    id: str
    chunk_text: str
    metadata: ChunkMetadata
    distance: float


def chunk_metadata(
    *,
    document_id: UUID,
    title: str,
    source: str | None,
    document_type: str,
    tags: tuple[str, ...],
    ordinal: int,
) -> ChunkMetadata:
    """Design §2.3's per-chunk metadata. Scalars only, so no list appears here.

    `source` is omitted rather than sent as `None`. Chroma treats a null value
    as deleting the key, so a document without a source has no key at all — and
    a `where` clause on `source` then correctly fails to match it, instead of
    matching a string that says "None".
    """
    metadata: ChunkMetadata = {
        DOCUMENT_ID_KEY: str(document_id),
        "title": title,
        "type": document_type,
        "tags": render_tags(tags),
        "ordinal": ordinal,
    }
    if source is not None:
        metadata["source"] = source
    return metadata


@dataclass(frozen=True, slots=True)
class MatchMetadata:
    """What a search hit says about the chunk it came from (PRD §6.2).

    The reader for what `chunk_metadata` writes, and it lives beside the writer
    so the two cannot drift. AD-004 builds the response from the Chroma payload
    alone, which makes this the only place the metadata is interpreted.

    Every field is defaulted because the payload comes back from a store, not
    from a schema. A vector written by an earlier build, or a collection an
    operator has poked at, is a wrong answer to give a caller — not a `500` to
    raise in the middle of an otherwise good result set.
    """

    document_id: str = ""
    title: str = ""
    type: str = ""
    tags: tuple[str, ...] = ()
    source: str | None = None
    ordinal: int = 0


def read_metadata(metadata: ChunkMetadata) -> MatchMetadata:
    """Undo `chunk_metadata`, turning the pipe-delimited tags back into a list."""
    source = metadata.get("source")
    ordinal = metadata.get("ordinal")
    return MatchMetadata(
        document_id=str(metadata.get(DOCUMENT_ID_KEY, "")),
        title=str(metadata.get("title", "")),
        type=str(metadata.get("type", "")),
        tags=parse_tags(str(metadata.get("tags", ""))),
        source=str(source) if source is not None else None,
        ordinal=int(ordinal) if isinstance(ordinal, int | float) else 0,
    )
