"""Batched embedding, shared by ingestion and reconciliation.

Design §3.1 step 5 caps a request at 96 inputs or 100 000 tokens, whichever
binds first, and `ai_embeddings.batch_texts` already implements that split. What
is left is the part both callers get wrong the same way if each writes it: the
provider rejects an oversized batch at the port rather than truncating it, so
skipping the split turns a 40-chunk document into a `422` instead of two
requests.

Returned as a flat tuple in the order the texts were given, plus the totals the
audit trail wants — tokens spent and how many API calls it took.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ai_embeddings import EmbeddingProvider, EmbeddingVector, batch_texts

__all__ = ["EmbeddedTexts", "embed_in_batches"]


@dataclass(frozen=True, slots=True)
class EmbeddedTexts:
    vectors: tuple[EmbeddingVector, ...]
    tokens: int
    calls: int


async def embed_in_batches(provider: EmbeddingProvider, texts: Sequence[str]) -> EmbeddedTexts:
    """Embed `texts`, split into request-sized batches, order preserved."""
    if not texts:
        return EmbeddedTexts((), 0, 0)

    model = provider.model
    batches = batch_texts(
        texts,
        max_inputs=model.max_batch_inputs,
        max_tokens=provider.max_batch_tokens,
        encoding=model.encoding,
    )

    vectors: list[EmbeddingVector] = []
    tokens = 0
    for batch in batches:
        result = await provider.embed_documents(batch)
        vectors.extend(result.vectors)
        tokens += result.tokens
    return EmbeddedTexts(tuple(vectors), tokens, len(batches))
