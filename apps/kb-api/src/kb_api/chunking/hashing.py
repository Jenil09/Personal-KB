"""The AD-008 hashes and the deterministic chunk identity.

Three derivations, all pure functions of already-normalised text:

- `content_hash(content)` — the document idempotency key. Same hash in the same
  collection means "nothing to do", so it must not depend on anything but the
  bytes.
- `text_hash(text, model_id)` — the carry-forward key. The model is part of it
  because a vector is only comparable to vectors from the same model (AD-006):
  identical text under a different model is a different row, not a reuse.
- `chunk_id(...)` / `chroma_id(...)` — identity. A UUID5 rather than a UUID4 so
  that retrying an interrupted ingest writes the same ids into both stores; a
  random id would make the Chroma upsert after a crash create a second copy of
  every vector instead of overwriting the first.
"""

import hashlib
from uuid import UUID, uuid5

__all__ = ["CHUNK_NAMESPACE", "chroma_id", "chunk_id", "content_hash", "text_hash"]

# A fixed, arbitrary namespace. Its only job is to keep these UUIDs from
# colliding with UUID5s derived elsewhere from the same strings.
CHUNK_NAMESPACE = UUID("6b3f2b3a-1d94-5f4c-9e3a-0a5c9b7d2e11")


def content_hash(content: str) -> str:
    """sha256 of normalised document content."""
    return hashlib.sha256(content.encode()).hexdigest()


def text_hash(text: str, model_id: str) -> str:
    """sha256 of normalised chunk text plus the model that will embed it.

    The separator keeps `("ab", "cd")` and `("a", "bcd")` apart; a newline
    cannot appear in a model id, so the split point is unambiguous.
    """
    return hashlib.sha256(f"{text}\n{model_id}".encode()).hexdigest()


def chunk_id(document_id: UUID, ordinal: int, chunk_text_hash: str) -> UUID:
    """Stable id for one chunk of one document.

    Includes the hash as well as the ordinal so that re-ingesting an edited
    document gives changed chunks new ids rather than quietly rebinding an
    existing id to different text.
    """
    return uuid5(CHUNK_NAMESPACE, f"{document_id}:{ordinal}:{chunk_text_hash}")


def chroma_id(chunk_uuid: UUID) -> str:
    """The Chroma-side primary key. Same value, string-typed — Chroma ids are
    strings, and deriving one from the other keeps the two stores joinable
    without carrying a second identifier."""
    return str(chunk_uuid)
