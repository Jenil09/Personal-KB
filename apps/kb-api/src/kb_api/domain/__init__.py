"""Entities and value objects. No framework, no I/O, no SQLAlchemy.

Ports live here too, per the layering rule in ai-kb/TECHNICAL-DESIGN.md §1 —
but none are defined yet. The repositories in `adapters/postgres` have no
service consuming them until Phase 6, and a `Protocol` written before its caller
exists describes a guess rather than a requirement. It gets extracted when the
ingestion service binds to it.
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

__all__ = [
    "Chunk",
    "Document",
    "DocumentFilter",
    "DocumentPage",
    "DocumentStatus",
    "IpAddress",
    "NewChunk",
    "NewDocument",
    "collection_name",
    "model_slug",
]
