"""The `EmbeddingProvider` port every driver implements.

Two methods, not one with a flag (AD-007). Gemini's retrieval quality depends on
sending `RETRIEVAL_DOCUMENT` at ingest and `RETRIEVAL_QUERY` at search; OpenAI
has no equivalent. Folding that into a keyword argument makes it easy to forget,
and forgetting it degrades Gemini silently, with no error to notice. Two methods
make the distinction impossible to skip.

Validation lives on the base class rather than in each adapter. The contract
suite asserts that both drivers reject the same inputs the same way, and the only
way to actually guarantee that is for both to run the same code.

`EmbeddingModel` is the identity of an index. A collection is bound to exactly
one for its lifetime (AD-006): vectors from different models occupy unrelated
spaces, and matching dimensionality does not make them comparable.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from platform_core import UpstreamError, ValidationError

__all__ = [
    "Embedding",
    "EmbeddingBatch",
    "EmbeddingModel",
    "EmbeddingProvider",
    "EmbeddingVector",
    "Purpose",
    "RawEmbeddings",
    "TokenSource",
]

EmbeddingVector = tuple[float, ...]

_NON_SLUG = re.compile(r"[^a-z0-9]+")


class Purpose(StrEnum):
    """Which side of retrieval a text is on. Gemini's `task_type` (AD-007)."""

    DOCUMENT = "document"
    QUERY = "query"


class TokenSource(StrEnum):
    """Whether a token count is the provider's or our own estimate.

    OpenAI reports exact usage per request. Gemini's embedding response carries
    a billable *character* count and no token count, so its figure is tiktoken's
    estimate against a different tokeniser and is not a billing number. Callers
    writing token totals into the audit trail need to know which they have.
    """

    PROVIDER = "provider"
    ESTIMATED = "estimated"


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """What a collection is bound to for its lifetime (AD-006)."""

    provider: str
    model_id: str
    dimensions: int
    max_input_tokens: int
    max_batch_inputs: int
    # tiktoken encoding used for pre-flight budgeting. Exact for OpenAI, an
    # approximation for anything else.
    encoding: str = "cl100k_base"

    @property
    def slug(self) -> str:
        return _NON_SLUG.sub("_", self.model_id.lower()).strip("_")

    def overridden(self, model_id: str | None, dimensions: int | None) -> "EmbeddingModel":
        """Apply an operator's model or dimension override, if there is one.

        Both change the collection name, so an override addresses a different
        index rather than reinterpreting the existing one (AD-006).
        """
        if model_id is None and dimensions is None:
            return self
        return replace(
            self,
            model_id=model_id or self.model_id,
            dimensions=dimensions or self.dimensions,
        )

    def collection_name(self, chunker_version: int) -> str:
        """`kb__openai__text_embedding_3_small__1536__c1` (Design §2.3).

        The chunker version is in the name because a chunker change invalidates
        every hash and forces a re-embed (AD-008), and that has to land in a new
        collection rather than corrupt the existing one.
        """
        return f"kb__{self.provider}__{self.slug}__{self.dimensions}__c{chunker_version}"


@dataclass(frozen=True, slots=True)
class Embedding:
    """One vector, for one query."""

    vector: EmbeddingVector
    model: EmbeddingModel
    tokens: int
    token_source: TokenSource


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors in the order their texts were given, plus the batch's token cost."""

    vectors: tuple[EmbeddingVector, ...]
    model: EmbeddingModel
    tokens: int
    token_source: TokenSource

    def __len__(self) -> int:
        return len(self.vectors)


@dataclass(frozen=True, slots=True)
class RawEmbeddings:
    """An adapter's answer before the port checks its shape."""

    vectors: tuple[EmbeddingVector, ...]
    tokens: int
    token_source: TokenSource


class EmbeddingProvider(ABC):
    """The port. Adapters implement `model` and `_embed`; the rest is shared."""

    #: Batching ceiling from Design §3.1 step 5, whichever binds first.
    max_batch_tokens: ClassVar[int] = 100_000

    @property
    @abstractmethod
    def model(self) -> EmbeddingModel: ...

    @abstractmethod
    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        """Call the provider. Never called with an empty or invalid batch."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Pre-flight budgeting: how many tokens this text will cost to embed."""

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed content for indexing. Gemini sends `RETRIEVAL_DOCUMENT`."""
        texts = list(texts)
        if not texts:
            # An ingest whose chunks were all unchanged (AD-008) legitimately
            # has nothing to embed. That is a zero-cost success, not an error.
            return EmbeddingBatch((), self.model, 0, TokenSource.PROVIDER)
        self._validate(texts)
        return self._as_batch(await self._embed(texts, Purpose.DOCUMENT), expected=len(texts))

    async def embed_query(self, text: str) -> Embedding:
        """Embed a search query. Gemini sends `RETRIEVAL_QUERY`."""
        self._validate([text])
        raw = self._as_batch(await self._embed([text], Purpose.QUERY), expected=1)
        return Embedding(raw.vectors[0], self.model, raw.tokens, raw.token_source)

    def _validate(self, texts: Sequence[str]) -> None:
        """Reject what the provider would reject, before spending a request on it."""
        if len(texts) > self.model.max_batch_inputs:
            raise ValidationError(
                f"{self.model.model_id} accepts at most {self.model.max_batch_inputs} "
                f"inputs per request, got {len(texts)}",
                context={"inputs": len(texts)},
            )
        total = 0
        for index, text in enumerate(texts):
            if not text.strip():
                # Providers answer 400 for these, and a blank chunk has no
                # meaningful position in the vector space anyway.
                raise ValidationError(
                    f"input {index} is empty or whitespace", context={"index": index}
                )
            tokens = self.count_tokens(text)
            if tokens > self.model.max_input_tokens:
                raise ValidationError(
                    f"input {index} is {tokens} tokens, over {self.model.model_id}'s "
                    f"{self.model.max_input_tokens} limit",
                    context={"index": index, "tokens": tokens},
                )
            total += tokens
        if total > self.max_batch_tokens:
            raise ValidationError(
                f"batch is {total} tokens, over the {self.max_batch_tokens} per-request budget",
                context={"tokens": total},
            )

    def _as_batch(self, raw: RawEmbeddings, *, expected: int) -> EmbeddingBatch:
        """Check the provider returned what was asked for, in usable shape.

        Both failures here have been seen in the wild and neither raises on its
        own: a provider that silently drops an input leaves vectors misaligned
        with chunks, and a dimension mismatch is only noticed when Chroma
        rejects the upsert, long after the tokens are spent.
        """
        if len(raw.vectors) != expected:
            raise UpstreamError(
                f"{self.model.provider} returned {len(raw.vectors)} vectors for {expected} inputs",
                context={"expected": expected, "returned": len(raw.vectors)},
            )
        wrong = [len(vector) for vector in raw.vectors if len(vector) != self.model.dimensions]
        if wrong:
            raise UpstreamError(
                f"{self.model.provider} returned {wrong[0]}-dimension vectors, "
                f"expected {self.model.dimensions}",
                context={"expected": self.model.dimensions, "returned": wrong[0]},
            )
        return EmbeddingBatch(raw.vectors, self.model, raw.tokens, raw.token_source)
