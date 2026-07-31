"""Gemini driver — `gemini-embedding-001`, with the task types AD-007 exists for.

Available as the `ai-embeddings[gemini]` extra (AD-010). In v1 this is evaluation
tooling, migration machinery, and outage insurance rather than a live per-request
alternative: a collection is bound to one model, and the two models' vectors are
not comparable (AD-006).

Two things here differ from the OpenAI driver in ways that matter.

**`task_type` is set per method.** `RETRIEVAL_DOCUMENT` at ingest,
`RETRIEVAL_QUERY` at search. Getting this wrong does not error — it quietly
returns worse results, which is precisely why the port has two methods rather
than one with a flag.

**Token counts are estimates.** The embedding response reports a billable
*character* count and no token count, so the figure returned is tiktoken's
against OpenAI's encoder and is marked `ESTIMATED`. It is good enough for
pre-flight batching and must not be treated as a billing number.
"""

from collections.abc import Sequence
from http import HTTPStatus

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

from ai_embeddings.port import (
    EmbeddingModel,
    EmbeddingProvider,
    EmbeddingVector,
    Purpose,
    RawEmbeddings,
    TokenSource,
)
from ai_embeddings.settings import EmbeddingProviderSettings
from ai_embeddings.tokens import count_tokens, total_tokens
from platform_core import ConfigurationError, UpstreamError

__all__ = ["GEMINI_EMBEDDING_001", "GeminiEmbeddingProvider"]

# 1536 rather than the model's native 3072: it matches the v1 index width, and
# Matryoshka truncation is supported by the model. It does not make the two
# providers' vectors comparable — nothing does (AD-006).
GEMINI_EMBEDDING_001 = EmbeddingModel(
    provider="gemini",
    model_id="gemini-embedding-001",
    dimensions=1536,
    max_input_tokens=2048,
    max_batch_inputs=96,
    encoding="cl100k_base",
)

_TASK_TYPES = {
    Purpose.DOCUMENT: "RETRIEVAL_DOCUMENT",
    Purpose.QUERY: "RETRIEVAL_QUERY",
}

_NATIVE_DIMENSIONS = 3072


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        settings: EmbeddingProviderSettings,
        http_client: httpx.AsyncClient,
        *,
        model: EmbeddingModel = GEMINI_EMBEDDING_001,
    ) -> None:
        self._model = model.overridden(settings.model_id, settings.dimensions)
        self._client = genai.Client(
            api_key=settings.api_key.get_secret_value(),
            # Passing the client also opts the SDK out of aiohttp, so the
            # service's own timeouts and connection pool are the ones in force.
            http_options=types.HttpOptions(
                httpx_async_client=http_client,
                retry_options=types.HttpRetryOptions(attempts=settings.max_retries + 1),
            ),
        )

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    def count_tokens(self, text: str) -> int:
        """An estimate, on OpenAI's encoder. See the module docstring."""
        return count_tokens(text, self._model.encoding)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        try:
            response = await self._client.aio.models.embed_content(
                model=self._model.model_id,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    task_type=_TASK_TYPES[purpose],
                    output_dimensionality=self._model.dimensions,
                ),
            )
        except APIError as exc:
            raise _from_api_error(exc) from exc

        embeddings = response.embeddings or []
        vectors: tuple[EmbeddingVector, ...] = tuple(
            _normalise(tuple(embedding.values or ()), self._model.dimensions)
            for embedding in embeddings
        )
        return RawEmbeddings(
            vectors, total_tokens(list(texts), self._model.encoding), TokenSource.ESTIMATED
        )


def _normalise(vector: EmbeddingVector, dimensions: int) -> EmbeddingVector:
    """Re-normalise a truncated vector to unit length.

    Only the full 3072-dimension output is normalised by the model. A truncated
    one is not, and the collection is cosine-space (Design §2.3), so leaving it
    unnormalised distorts every distance it takes part in.
    """
    if dimensions >= _NATIVE_DIMENSIONS or not vector:
        return vector
    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return vector
    return tuple(value / magnitude for value in vector)


def _from_api_error(exc: APIError) -> Exception:
    """A rejected key is a misconfiguration; everything else is an outage."""
    if exc.code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return ConfigurationError(f"gemini rejected the API key: {exc.code}")
    return UpstreamError(
        f"gemini embeddings failed with {exc.code}",
        context={"status_code": exc.code},
    )
