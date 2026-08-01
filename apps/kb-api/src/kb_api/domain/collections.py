"""Collection naming (Design §2.3).

`kb__{provider}__{model_slug}__{dims}__c{chunker_version}`, e.g.
`kb__openai__text_embedding_3_small__1536__c1`.

Every component is something that makes vectors incomparable when it changes:
the provider and model decide the embedding space (AD-006), the dimension count
decides the shape, and the chunker version decides the text that was embedded
(AD-008). Encoding all four in the name means a mismatch is a missing
collection — a `409` at ingest — rather than a search that silently returns
nonsense from a space it does not belong to.

`chunker_version` is passed in rather than read from `kb_api.chunking` so the
domain layer stays free of its implementation; the composition root supplies
`CHUNKER_VERSION`.
"""

import re

__all__ = ["collection_name", "model_slug"]

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def model_slug(model_id: str) -> str:
    """`text-embedding-3-small` → `text_embedding_3_small`.

    Chroma restricts collection names, and provider model ids carry dots and
    slashes (`models/gemini-embedding-001`), so everything outside
    `[a-z0-9_]` collapses to a single underscore.
    """
    slug = _NON_SLUG.sub("_", model_id.lower()).strip("_")
    if not slug:
        raise ValueError(f"model id has no usable characters: {model_id!r}")
    return slug


def collection_name(*, provider: str, model_id: str, dimensions: int, chunker_version: int) -> str:
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    if chunker_version < 1:
        raise ValueError("chunker_version must be positive")
    return f"kb__{model_slug(provider)}__{model_slug(model_id)}__{dimensions}__c{chunker_version}"
