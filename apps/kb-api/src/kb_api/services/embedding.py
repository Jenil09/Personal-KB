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
    token_source: str = "exact"
    """Where the count came from (AD-017).

    Carried alongside the number rather than derived from the provider name by
    the reader, because the whole point of AD-017 is that an estimate must not
    be mistaken for a billing figure — and it would be, the first time someone
    summed this column without checking which provider produced each row.

    Degraded across a multi-batch call: if any batch was estimated the total is
    estimated, since a mixed total is not exact.
    """


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
    sources: set[str] = set()
    for batch in batches:
        result = await provider.embed_documents(batch)
        vectors.extend(result.vectors)
        tokens += result.tokens
        sources.add(str(result.token_source))
    source = sources.pop() if len(sources) == 1 else "estimated"
    return EmbeddedTexts(tuple(vectors), tokens, len(batches), source)
