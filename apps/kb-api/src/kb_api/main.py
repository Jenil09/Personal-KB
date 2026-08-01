"""The composition root — the only module that knows which adapter is which.

Everything above this file talks to ports. Here the concrete
`DocumentRepository`, `ChunkRepository`, and `ChromaVectorStore` are bound to
them, which is what makes Design §1's layering rule hold rather than merely be
described.

The whole object graph is built synchronously in `build_app`, before the app
exists, so routers are mounted with the service they call already in hand. The
lifespan handler only opens and closes connections. The alternative — building
services inside the lifespan and reaching for them through `app.state` from a
handler — turns the composition root into a service locator and puts the
knowledge of what is bound back into the routers.

Startup order in the lifespan handler is not arbitrary:

1. Open Chroma. It is the one client whose construction is a network call.
2. Ping Postgres, so a bad DSN fails here rather than on the first request.
3. Run reconciliation, *before* the app serves. A `pending` document left by a
   crash should be repaired before a search can miss it, and a soft-deleted
   document whose vectors survived should stop being answerable before anything
   asks (`services/reconciliation.py`).
4. On shutdown, close the provider HTTP clients and dispose the engine.

Reconciliation does not block startup on failure; it reports and logs. A service
that refuses to start over one unrepairable row leaves the operator no endpoint
to ask about it from.

`build_app` is a factory rather than a module-level `app`, and uvicorn is run
with `--factory`. A module-level instance would read the environment at import
time, which makes the module unimportable without a full configuration — so a
test that wants an app with stubbed ports could not import the function that
builds one.

The `/health` checks here are Postgres and Chroma only. Queue depth, spill-file
state, and the tier-1 audit wiring are Phase 8's — this is the subset that has
something to check as of Phase 6.
"""

from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from ai_embeddings import EmbeddingProvider, create_http_client
from kb_api.adapters.chroma import ChromaVectorStore, create_chroma_client
from kb_api.adapters.postgres import ChunkRepository, DocumentRepository
from kb_api.api.v1 import create_documents_router
from kb_api.chunking import CHUNKER_VERSION
from kb_api.config import KbApiSettings, get_settings
from kb_api.services import IngestionService, ProviderRegistry, ReconciliationService
from platform_core import ConfigurationError, get_logger
from platform_db import Database
from platform_fastapi import CheckResult, HealthCheck, create_app

__all__ = ["build_app"]

_logger = get_logger("kb.startup")


def build_app(
    settings: KbApiSettings | None = None,
    *,
    providers: Mapping[str, EmbeddingProvider] | None = None,
) -> FastAPI:
    """Assemble the service.

    `providers` overrides the drivers built from configuration. Design §7 runs
    E2E tests through `httpx.ASGITransport` with the embedding providers stubbed
    *at the port* — this is that seam, and having it here means a test exercises
    the real routers, the real services, and the real adapters, with only the
    one dependency that costs money and network replaced.
    """
    resolved = settings or get_settings()

    database = Database(resolved.postgres)
    if providers is None:
        drivers, http_clients = _build_providers(resolved)
    else:
        drivers, http_clients = dict(providers), []
    registry = ProviderRegistry(
        drivers, default=resolved.default_provider, chunker_version=CHUNKER_VERSION
    )
    vectors = ChromaVectorStore(
        lambda: create_chroma_client(
            host=resolved.chroma.host,
            port=resolved.chroma.port,
            ssl=resolved.chroma.ssl,
            tenant=resolved.chroma.tenant,
            database=resolved.chroma.database,
        )
    )
    documents = DocumentRepository()
    chunks = ChunkRepository()

    ingestion = IngestionService(
        sessions=database,
        documents=documents,
        chunks=chunks,
        vectors=vectors,
        providers=registry,
    )
    reconciliation = ReconciliationService(
        sessions=database,
        documents=documents,
        chunks=chunks,
        vectors=vectors,
        providers=registry,
        batch_limit=resolved.reconciliation_limit,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await vectors.connect()
        if not await database.ping():
            raise ConfigurationError("Postgres is not reachable; check KB_API__POSTGRES__DSN")

        report = await reconciliation.run()
        _logger.info(
            "startup_complete",
            collections=[registry.resolve(name).collection for name in registry.names],
            reindexed=report.reindexed,
            purged=report.purged,
            failed=report.failed,
        )
        try:
            yield
        finally:
            for client in http_clients:
                await client.aclose()
            await database.dispose()

    return create_app(
        resolved,
        routers=(create_documents_router(ingestion),),
        lifespan=lifespan,
        health_checks=(
            HealthCheck("postgres", lambda: _probe(database.ping())),
            HealthCheck("chromadb", lambda: _probe(vectors.heartbeat())),
        ),
    )


def _build_providers(
    settings: KbApiSettings,
) -> tuple[dict[str, EmbeddingProvider], list[httpx.AsyncClient]]:
    """One driver per configured key, each with its own pooled HTTP client.

    The drivers are imported inside the function because they are optional
    extras (AD-010): a deployment that configures only OpenAI must not need the
    Gemini SDK installed in order to start.

    The clients are returned rather than closed here — they live as long as the
    process and are closed in the lifespan handler (Design §5). A per-request
    client would pay a TLS handshake on every search.
    """
    providers: dict[str, EmbeddingProvider] = {}
    clients: list[httpx.AsyncClient] = []

    if settings.openai is not None:
        from ai_embeddings.adapters.openai import OpenAIEmbeddingProvider

        client = create_http_client(settings.openai)
        clients.append(client)
        providers["openai"] = OpenAIEmbeddingProvider(settings.openai, client)

    if settings.gemini is not None:
        from ai_embeddings.adapters.gemini import GeminiEmbeddingProvider

        client = create_http_client(settings.gemini)
        clients.append(client)
        providers["gemini"] = GeminiEmbeddingProvider(settings.gemini, client)

    if not providers:
        raise ConfigurationError(
            "no embedding provider is configured; set KB_API__OPENAI__API_KEY "
            "or KB_API__GEMINI__API_KEY"
        )
    return providers, clients


async def _probe(check: Awaitable[bool]) -> CheckResult:
    """Adapt a boolean liveness check to the health router's verdict type."""
    return CheckResult.ok() if await check else CheckResult.unavailable()
