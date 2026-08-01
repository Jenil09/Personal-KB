"""Entities, value objects, and the ports the services bind to.

No I/O happens here and no adapter is imported. One framework type does appear:
`ports.py` names `AsyncSession`, because the repository contract *is* "do this
inside the caller's transaction" and there is no way to say that without naming
the thing the transaction lives on. Typing it away as `Any` would delete the
only part of the signature worth checking, and handing services pre-bound
repositories would move the transaction boundary out of the flow that decides
what belongs in one. Nothing else in this layer imports SQLAlchemy.

The ports arrived in Phase 6 rather than Phase 4 on purpose: a `Protocol`
written before its caller describes a guess, so each one names exactly the
methods the ingestion and reconciliation flows call and nothing more.
"""

from kb_api.domain.collections import collection_name, model_slug
from kb_api.domain.documents import (
    Chunk,
    Document,
    DocumentFilter,
    DocumentPage,
    DocumentStatus,
    IpAddress,
    NewChunk,
    NewDocument,
)
from kb_api.domain.ports import ChunkStore, DocumentStore, TelemetryPort, VectorStore
from kb_api.domain.vectors import (
    DOCUMENT_ID_KEY,
    TAG_SEPARATOR,
    ChunkMetadata,
    MatchMetadata,
    VectorMatch,
    VectorRecord,
    chunk_metadata,
    parse_tags,
    read_metadata,
    render_tags,
)

__all__ = [
    "DOCUMENT_ID_KEY",
    "TAG_SEPARATOR",
    "Chunk",
    "ChunkMetadata",
    "ChunkStore",
    "Document",
    "DocumentFilter",
    "DocumentPage",
    "DocumentStatus",
    "DocumentStore",
    "IpAddress",
    "MatchMetadata",
    "NewChunk",
    "NewDocument",
    "TelemetryPort",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
    "chunk_metadata",
    "collection_name",
    "model_slug",
    "parse_tags",
    "read_metadata",
    "render_tags",
]
