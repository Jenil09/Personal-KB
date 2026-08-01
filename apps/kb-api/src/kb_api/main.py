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

`/health` reports four things: Postgres, Chroma, the tier-2 queue, and the tier-1
spill file. The last two are the ones with three-valued answers — a non-empty
spill is `degraded` and still `200`, because the service is serving correctly and
what needs an operator is the durability of the trail, not the request path
(AD-013). Taking the instance out of rotation over it would turn a Postgres blip
into an outage.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from traceback import format_exception

import httpx
from fastapi import FastAPI

from ai_embeddings import EmbeddingProvider, create_http_client
from kb_api.adapters.chroma import ChromaVectorStore, create_chroma_client
from kb_api.adapters.postgres import ChunkRepository, DocumentRepository, TelemetryRecorder
from kb_api.api.v1 import create_admin_router, create_documents_router, create_search_router
from kb_api.chunking import CHUNKER_VERSION
from kb_api.config import KbApiSettings, get_settings
from kb_api.services import (
    DocumentService,
    IngestionService,
    ProviderRegistry,
    QueryEmbeddingCache,
    ReconciliationService,
    SearchService,
    StatsService,
)
from platform_core import ConfigurationError, get_logger
from platform_db import AuditRecord, AuditTrail, Database, Outcome, SpillFile, TelemetrySink
from platform_fastapi import CheckResult, HealthCheck, HealthStatus, create_app

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

    # Tier 1 and tier 2 (AD-013). The trail is handed to `create_app` as
    # middleware; the sink is handed to the flows that emit. Both are started
    # and stopped in the lifespan handler, because both own a background task.
    trail = AuditTrail(database, SpillFile(resolved.audit.spill_path))
    sink = TelemetrySink(database, resolved.audit)
    telemetry = TelemetryRecorder(sink)

    ingestion = IngestionService(
        sessions=database,
        documents=documents,
        chunks=chunks,
        vectors=vectors,
        providers=registry,
        telemetry=telemetry,
    )
    document_management = DocumentService(
        sessions=database,
        documents=documents,
        chunks=chunks,
        vectors=vectors,
    )
    # One cache for the process, held by the service rather than module-level:
    # a single Uvicorn worker (AD-015) means one instance either way, and an
    # instance attribute is one a test can replace without reaching into a
    # module. Nothing in it survives a restart, by design.
    search = SearchService(
        sessions=database,
        documents=documents,
        vectors=vectors,
        providers=registry,
        cache=QueryEmbeddingCache(
            max_entries=resolved.query_cache_size,
            ttl_seconds=resolved.query_cache_ttl_seconds,
        ),
        tag_filter_limit=resolved.tag_filter_limit,
        telemetry=telemetry,
    )
    stats = StatsService(
        sessions=database,
        vectors=vectors,
        providers=registry,
        telemetry=sink,
        trail=trail,
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

        # Before the first request, so a spill left by the last outage is
        # already draining while the service starts answering.
        await trail.start()
        await sink.start()

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
            # Tier 2 first: its flush wants a live engine. Tier 1 needs no
            # draining here — anything it could not write is already on disk.
            await sink.stop()
            await trail.stop()
            for client in http_clients:
                await client.aclose()
            await database.dispose()

    return create_app(
        resolved,
        routers=(
            create_search_router(search),
            create_documents_router(ingestion, document_management),
            create_admin_router(stats),
        ),
        lifespan=lifespan,
        health_checks=(
            HealthCheck("postgres", lambda: _probe(database.ping())),
            HealthCheck("chromadb", lambda: _probe(vectors.heartbeat())),
            HealthCheck("audit_spill", lambda: _spill_check(trail)),
            HealthCheck(
                "telemetry_queue",
                lambda: _queue_check(sink, resolved.audit.telemetry_queue_size),
            ),
        ),
        audit_trail=trail,
        audit_observer=_error_observer(telemetry),
        rate_limits=resolved.rate_limits,
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


def _error_observer(
    telemetry: TelemetryRecorder,
) -> Callable[[AuditRecord, BaseException | None], None]:
    """One `error_logs` row per failed request, hung off the tier-1 write.

    The audit middleware has already decided what the outcome was and already
    holds the exception, so this reads that decision rather than making it a
    second time — which is what keeps the two tiers from disagreeing about
    whether a given request failed.
    """

    def observe(record: AuditRecord, failure: BaseException | None) -> None:
        if record.outcome in (Outcome.SUCCESS, Outcome.AUTH_FAILED):
            # A rejected key is not an error worth a stack trace; the tier-1 row
            # already records it, with the credential fingerprinted.
            return
        telemetry.error_occurred(
            request_id=record.request_id,
            error_code=record.error_code or "unknown",
            exception_type=type(failure).__name__ if failure else "HTTPResponse",
            message=str(failure) if failure else f"{record.status_code} on {record.path}",
            stack="".join(format_exception(failure)) if failure else None,
            context={"path": record.path, "method": record.method, "key_id": record.key_id},
        )

    return observe


async def _spill_check(trail: AuditTrail) -> CheckResult:
    """A non-empty tier-1 spill is `degraded`, never `unavailable`.

    The distinction is the reason health has three states rather than two. Records
    in the spill mean Postgres was unreachable when they were written and the
    reconciler has not drained them yet — the request path is fine, the trail's
    durability is not, and that wants an operator rather than a restart. A `503`
    here would take a healthy instance out of rotation over a backlog that is
    already being retried every thirty seconds.
    """
    depth = await trail.spill_depth()
    return CheckResult.ok("empty") if depth == 0 else CheckResult.degraded(f"{depth} pending")


async def _queue_check(sink: TelemetrySink, capacity: int) -> CheckResult:
    """Tier-2 saturation, reported before it becomes silent loss.

    Dropped records are the signal that matters — they are already lost — but
    depth is the one that arrives first, so both are reported and a queue over
    80% full is `degraded` while it is still absorbing.
    """
    depth = sink.depth
    detail = f"{depth}/{capacity}, {sink.dropped} dropped"
    saturated = depth >= capacity * 0.8
    return CheckResult(
        HealthStatus.DEGRADED if saturated or sink.dropped else HealthStatus.OK, detail
    )
