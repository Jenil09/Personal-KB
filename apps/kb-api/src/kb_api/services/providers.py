"""Provider name → the index it addresses (AD-006).

`provider` in a request body is not a choice of *how* to embed, it is a choice
of *which collection to talk to*. One model defines one index for its lifetime;
naming a provider selects the index bound to that model, and the two are the
same decision spelled two ways.

Which providers exist is a deployment fact — a key is configured or it is not —
so the registry is built in the composition root and this module never reads
settings. That also means the set of valid `provider` values cannot be an enum
in the API schema: it depends on what the operator configured.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from ai_embeddings import EmbeddingProvider
from kb_api.domain import collection_name
from platform_core import ValidationError

__all__ = ["ProviderRegistry", "ResolvedProvider"]


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    """A driver and the one collection its vectors belong in."""

    name: str
    provider: EmbeddingProvider
    collection: str


class ProviderRegistry:
    """Resolves a request's `provider` to a driver and a collection name."""

    def __init__(
        self, providers: Mapping[str, EmbeddingProvider], *, default: str, chunker_version: int
    ) -> None:
        if not providers:
            raise ValueError("at least one embedding provider must be configured")
        if default not in providers:
            raise ValueError(f"default provider {default!r} is not configured")
        self._default = default
        self._resolved = {
            name: ResolvedProvider(
                name=name,
                provider=provider,
                # `collection_name` rather than `EmbeddingModel.collection_name`:
                # the domain function slugifies the provider segment too, so a
                # driver registered under a name with a hyphen in it cannot
                # produce a collection name Chroma rejects.
                collection=collection_name(
                    provider=name,
                    model_id=provider.model.model_id,
                    dimensions=provider.model.dimensions,
                    chunker_version=chunker_version,
                ),
            )
            for name, provider in providers.items()
        }

    @property
    def default(self) -> ResolvedProvider:
        return self._resolved[self._default]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._resolved)

    def resolve(self, name: str | None) -> ResolvedProvider:
        """Look up a provider, falling back to the configured default.

        An unconfigured name is a `422`, not the `409` search raises for an
        unpopulated collection (Design §3.2 step 1). They are different
        failures: one says the value you sent is not a thing, the other says the
        thing exists but has nothing in it, and answering both the same way
        would make an operator's missing API key look like an empty corpus.
        """
        if name is None:
            return self.default
        resolved = self._resolved.get(name)
        if resolved is None:
            raise ValidationError(
                f"Unknown embedding provider {name!r}.",
                context={"provider": name, "configured": list(self._resolved)},
            )
        return resolved
