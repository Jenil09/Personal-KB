"""The Chroma adapter — the only module in the service that imports `chromadb`.

AD-010 keeps this inside `apps/kb-api` rather than in `libs/`: there is no
second consumer, and extracting a vector-store library for one caller invents
the seam before anything has pushed on it. The `VectorStore` port is what makes
that reversible, so nothing here leaks upward — not the client, not
`AsyncCollection`, not `QueryResult`, not `chromadb.errors`, and not the numpy
arrays `get` hands back.

Four things about Chroma shape the code below.

**Collections are handles, and resolving one is a round trip.** `get_collection`
is an HTTP call, so the obvious implementation pays one per operation. Handles
are cached by name. A collection deleted out from under a running process is an
operator action, not a request-path concern.

**The default embedding function is not wanted.** Chroma will embed text for you
with a bundled ONNX model unless told otherwise. Every call passes
`embedding_function=None`: the vectors come from the provider AD-006 bound the
collection to, and silently falling back to a different model is exactly the
failure that decision exists to prevent. The thin client does not ship that
model either, so the fallback would fail on import rather than politely.

**Failures are not all `ChromaError`.** A server that is down produces an
`httpx` transport error, which has nothing to do with Chroma's own exception
tree. Catching only `ChromaError` would let a connection refusal reach the
catch-all as an unhandled `500` instead of the `502` an unreachable dependency
warrants.

**Metadata values must be scalars** (AD-005). Nothing here encodes a list.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast
from uuid import UUID

import chromadb
import httpx
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.errors import ChromaError, NotFoundError

from ai_embeddings import EmbeddingVector
from kb_api.domain import DOCUMENT_ID_KEY, ChunkMetadata, VectorMatch, VectorRecord
from platform_core import ConfigurationError, ConflictError, UpstreamError, get_logger

__all__ = ["ChromaVectorStore", "create_chroma_client"]

_logger = get_logger("kb.chroma")

# Design §2.3. Chroma reads the space out of `metadata` at creation and reports
# it back there, which is what makes an existing collection's space checkable.
_COSINE = {"hnsw:space": "cosine"}

# Everything that means "the vector store did not answer". `NotFoundError` is a
# `ChromaError` too, so it is always caught first where it is meaningful.
_FAILURES = (ChromaError, httpx.HTTPError, OSError)


async def create_chroma_client(
    *, host: str, port: int, ssl: bool = False, tenant: str, database: str
) -> AsyncClientAPI:
    """Open the one client the process gets. Built in the lifespan handler."""
    return await chromadb.AsyncHttpClient(
        host=host, port=port, ssl=ssl, tenant=tenant, database=database
    )


class ChromaVectorStore:
    """`VectorStore` over a remote Chroma server.

    Takes a factory rather than a client because opening one is `async` —
    Chroma validates the tenant and database over the network before handing it
    back. Constructing the store synchronously is what lets the whole object
    graph, routers included, be assembled in `build_app`; the lifespan handler
    then only has to `connect()`. The alternative is binding routers to services
    that do not exist yet at construction time, which is how composition roots
    turn into service locators.
    """

    def __init__(self, connect: Callable[[], Awaitable[AsyncClientAPI]]) -> None:
        self._connect = connect
        self._client: AsyncClientAPI | None = None
        self._collections: dict[str, AsyncCollection] = {}

    async def connect(self) -> None:
        """Open the one client the process gets. Called from the lifespan handler."""
        self._client = await self._connect()

    @property
    def _api(self) -> AsyncClientAPI:
        if self._client is None:
            raise ConfigurationError("chroma client was never connected")
        return self._client

    async def ensure_collection(self, name: str) -> None:
        if name in self._collections:
            return
        try:
            collection = await self._api.get_or_create_collection(
                name=name, metadata=_COSINE, embedding_function=None
            )
        except _FAILURES as exc:
            raise _upstream("create collection", name, exc) from exc
        _check_space(collection)
        self._collections[name] = collection

    async def collection_exists(self, name: str) -> bool:
        return await self._resolve_optional(name) is not None

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        if not records:
            # A re-ingest where every chunk was carried forward has nothing to
            # write, and Chroma answers an empty upsert with a 400.
            return
        handle = await self._resolve(collection)
        try:
            await handle.upsert(
                ids=[record.id for record in records],
                # Annotated because Chroma's parameter is a union of
                # sequence types, and mypy will not widen `list[list[float]]`
                # into it on its own.
                embeddings=cast("list[Sequence[float]]", [list(r.vector) for r in records]),
                documents=[record.chunk_text for record in records],
                metadatas=[dict(record.metadata) for record in records],
            )
        except _FAILURES as exc:
            raise _upstream("upsert", collection, exc) from exc

    async def fetch_vectors(
        self, collection: str, ids: Sequence[str]
    ) -> Mapping[str, EmbeddingVector]:
        if not ids:
            return {}
        handle = await self._resolve(collection)
        try:
            result = await handle.get(ids=list(ids), include=["embeddings"])
        except _FAILURES as exc:
            raise _upstream("get", collection, exc) from exc

        embeddings = result.get("embeddings")
        if embeddings is None:
            return {}
        # Chroma hands back numpy arrays. One leaving this module would make
        # every consumer's type quietly depend on which vector store is bound.
        return {
            found: tuple(float(value) for value in vector)
            for found, vector in zip(result["ids"], embeddings, strict=True)
        }

    async def delete_document(self, collection: str, document_id: UUID) -> int:
        handle = await self._resolve_optional(collection)
        if handle is None:
            # Nothing to purge in a collection that was never created. Delete is
            # idempotent (Design §3.3), so this is a success with zero rows.
            return 0
        try:
            result = await handle.delete(where={DOCUMENT_ID_KEY: str(document_id)})
        except _FAILURES as exc:
            raise _upstream("delete", collection, exc) from exc
        return int(cast("Mapping[str, Any]", result).get("deleted", 0))

    async def query(
        self,
        collection: str,
        vector: EmbeddingVector,
        *,
        top_k: int,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorMatch, ...]:
        handle = await self._resolve(collection)
        try:
            result = await handle.query(
                query_embeddings=cast("list[Sequence[float]]", [list(vector)]),
                n_results=top_k,
                # `None` and `{}` are not the same to Chroma: an empty clause
                # matches nothing, which would silently empty every result set.
                where=cast("Any", dict(where)) if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except _FAILURES as exc:
            raise _upstream("query", collection, exc) from exc
        return _to_matches(result)

    async def count(self, collection: str) -> int:
        handle = await self._resolve_optional(collection)
        if handle is None:
            return 0
        try:
            return await handle.count()
        except _FAILURES as exc:
            raise _upstream("count", collection, exc) from exc

    async def heartbeat(self) -> bool:
        try:
            await self._api.heartbeat()
        except Exception as exc:
            _logger.warning("chroma_heartbeat_failed", exc_info=exc)
            return False
        return True

    async def _resolve(self, name: str) -> AsyncCollection:
        """The cached handle. A collection that is not there is a `409`.

        `ConflictError` rather than Chroma's own `NotFoundError`, and rather
        than a `404`: Design §3.2 answers a provider naming a collection with
        nothing in it with `409`, and "never created" is the same condition
        seen one moment earlier. Letting `chromadb.errors` out here would make
        it an unhandled `500` and would put the vector store's exception tree in
        the service's contract.
        """
        handle = await self._resolve_optional(name)
        if handle is None:
            raise ConflictError(
                f"collection {name} is not populated",
                context={"collection": name},
            )
        return handle

    async def _resolve_optional(self, name: str) -> AsyncCollection | None:
        """The cached handle, fetched once, or `None` if the collection is absent."""
        cached = self._collections.get(name)
        if cached is not None:
            return cached
        try:
            collection = await self._api.get_collection(name=name, embedding_function=None)
        except NotFoundError:
            return None
        except _FAILURES as exc:
            raise _upstream("get collection", name, exc) from exc
        self._collections[name] = collection
        return collection


def _check_space(collection: AsyncCollection) -> None:
    """Warn when an existing collection is not the cosine space §2.3 assumes.

    Chroma ignores the requested space on a collection that already exists, so
    reading back what came out is the only way to notice one created with the
    default L2 distance. Not fatal — the ordering is still a ranking — but
    `1 - distance` stops being a similarity, and an operator should hear about
    that rather than see negative scores in search results.
    """
    space = (collection.metadata or {}).get("hnsw:space")
    if space not in (None, "cosine"):
        _logger.warning("chroma_collection_space_mismatch", collection=collection.name, space=space)


def _to_matches(result: Mapping[str, Any]) -> tuple[VectorMatch, ...]:
    """Flatten Chroma's parallel lists into the port's records.

    Every field arrives as a list of lists — one inner list per query embedding —
    and any of them can be `None` when the matching `include` was omitted. One
    query goes in, so exactly one row comes back.
    """
    ids = (result.get("ids") or [[]])[0]
    if not ids:
        return ()
    return tuple(
        VectorMatch(
            id=str(identifier),
            chunk_text=text or "",
            metadata=cast("ChunkMetadata", dict(metadata or {})),
            distance=float(distance),
        )
        for identifier, text, metadata, distance in zip(
            ids,
            _row(result, "documents", len(ids), ""),
            _row(result, "metadatas", len(ids), {}),
            _row(result, "distances", len(ids), 0.0),
            strict=True,
        )
    )


def _row(result: Mapping[str, Any], key: str, width: int, fill: Any) -> list[Any]:
    """One query's slice of a Chroma result field, or `width` copies of `fill`."""
    outer = result.get(key)
    return list(outer[0]) if outer else [fill] * width


def _upstream(operation: str, collection: str, exc: Exception) -> UpstreamError:
    return UpstreamError(
        f"chroma {operation} failed on {collection}: {exc}",
        context={"collection": collection, "operation": operation},
    )
