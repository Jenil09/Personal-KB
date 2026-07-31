"""The `EmbeddingProvider` port, token accounting, and provider configuration.

Drivers live under `ai_embeddings.adapters` and are imported directly, so this
package can be imported without either provider SDK installed.
"""

from ai_embeddings.port import (
    Embedding,
    EmbeddingBatch,
    EmbeddingModel,
    EmbeddingProvider,
    EmbeddingVector,
    Purpose,
    RawEmbeddings,
    TokenSource,
)
from ai_embeddings.settings import EmbeddingProviderSettings, create_http_client
from ai_embeddings.tokens import batch_texts, count_tokens, encoding_for, total_tokens

__all__ = [
    "Embedding",
    "EmbeddingBatch",
    "EmbeddingModel",
    "EmbeddingProvider",
    "EmbeddingProviderSettings",
    "EmbeddingVector",
    "Purpose",
    "RawEmbeddings",
    "TokenSource",
    "__version__",
    "batch_texts",
    "count_tokens",
    "create_http_client",
    "encoding_for",
    "total_tokens",
]

__version__ = "0.1.0"
