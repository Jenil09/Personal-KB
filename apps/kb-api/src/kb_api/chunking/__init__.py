"""Deterministic Markdown chunking (Design §4, AD-008).

The public surface is `chunk_document`, the hashes, and the version constant
that binds this behaviour to a collection name. Everything else is stages of
the pipeline and is imported directly by the tests that exercise them.
"""

from kb_api.chunking.chunker import ChunkDraft, chunk_document, to_new_chunk
from kb_api.chunking.config import CHUNKER_VERSION, DEFAULT_CONFIG, ChunkerConfig
from kb_api.chunking.hashing import chroma_id, chunk_id, content_hash, text_hash
from kb_api.chunking.normalise import normalise

__all__ = [
    "CHUNKER_VERSION",
    "DEFAULT_CONFIG",
    "ChunkDraft",
    "ChunkerConfig",
    "chroma_id",
    "chunk_document",
    "chunk_id",
    "content_hash",
    "normalise",
    "text_hash",
    "to_new_chunk",
]
