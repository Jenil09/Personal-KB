"""OpenAI driver — `text-embedding-3-small`, the model the v1 index is bound to.

Available as the `ai-embeddings[openai]` extra (AD-010), so a service that only
uses Gemini does not carry this SDK.

OpenAI has no task type, so `embed_documents` and `embed_query` reach the API
identically. The asymmetry AD-007 asks for still exists at the port; here it is
simply a no-op, which is the correct implementation of it rather than a missing
one.
"""

from collections.abc import Sequence
from http import HTTPStatus

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI

from ai_embeddings.port import (
    EmbeddingModel,
    EmbeddingProvider,
    EmbeddingVector,
    Purpose,
    RawEmbeddings,
    TokenSource,
)
from ai_embeddings.settings import EmbeddingProviderSettings
from ai_embeddings.tokens import count_tokens
from platform_core import ConfigurationError, UpstreamError

__all__ = ["TEXT_EMBEDDING_3_SMALL", "OpenAIEmbeddingProvider"]

TEXT_EMBEDDING_3_SMALL = EmbeddingModel(
    provider="openai",
    model_id="text-embedding-3-small",
    dimensions=1536,
    max_input_tokens=8191,
    # The API's own ceiling on inputs per request; Design §3.1 batches to it.
    max_batch_inputs=96,
    encoding="cl100k_base",
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        settings: EmbeddingProviderSettings,
        http_client: httpx.AsyncClient,
        *,
        model: EmbeddingModel = TEXT_EMBEDDING_3_SMALL,
    ) -> None:
        self._model = model.overridden(settings.model_id, settings.dimensions)
        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            http_client=http_client,
            max_retries=settings.max_retries,
        )

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    def count_tokens(self, text: str) -> int:
        # cl100k_base is the encoder the API bills with, so this figure and
        # `usage.prompt_tokens` agree exactly.
        return count_tokens(text, self._model.encoding)

    async def _embed(self, texts: Sequence[str], purpose: Purpose) -> RawEmbeddings:
        try:
            response = await self._client.embeddings.create(
                model=self._model.model_id,
                input=list(texts),
                dimensions=self._model.dimensions,
                encoding_format="float",
            )
        except APIStatusError as exc:
            raise _from_status(exc) from exc
        except APIError as exc:
            raise UpstreamError(f"openai embeddings failed: {exc}") from exc

        # The API documents order as unspecified and returns an `index` on each
        # item; sorting is what keeps vectors aligned with the chunks they came
        # from, and misalignment is silent corruption of the index.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors: tuple[EmbeddingVector, ...] = tuple(tuple(item.embedding) for item in ordered)
        return RawEmbeddings(vectors, response.usage.prompt_tokens, TokenSource.PROVIDER)


def _from_status(exc: APIStatusError) -> Exception:
    """A rejected key is a misconfiguration; everything else is an outage."""
    if exc.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return ConfigurationError(f"openai rejected the API key: {exc.status_code}")
    return UpstreamError(
        f"openai embeddings failed with {exc.status_code}",
        context={"status_code": exc.status_code},
    )
