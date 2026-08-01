"""The admin statistics endpoint's data (PRD §6.7).

Three sources, and the split between them is the point of the endpoint. Postgres
answers what the corpus *is* — documents, chunks, tokens stored. Chroma answers
what the index *holds*, per collection. The audit machinery answers whether the
observability itself is healthy: how many tier-2 records were dropped, how deep
the tier-1 spill is, how many bursts were flagged.

That third group is why this is worth an endpoint rather than a log line. A
dropped telemetry record and a spilled audit record are both invisible by
construction — the first is gone, the second is on disk where nothing reads it —
so the only way either becomes known is by being asked for.

**Corpus counts and index counts are reported side by side, never reconciled.**
Postgres is the system of record and Chroma is a rebuildable index (AD-003), so a
disagreement between them is a fact about the deployment worth seeing, not a
discrepancy to paper over by picking one. Chroma being unreachable reports as
`None` for that collection rather than failing the request; an operator asking
for stats during an outage is asking precisely because there is one.
"""

from dataclasses import dataclass, field

from kb_api.adapters.postgres.stats import CorpusStats, StatsRepository, TokenUsage
from kb_api.domain import VectorStore
from kb_api.services.providers import ProviderRegistry
from platform_core import get_logger
from platform_db import AuditTrail, SessionSource, TelemetrySink

__all__ = ["CollectionStats", "ServiceStats", "StatsService"]

_logger = get_logger("kb.stats")


@dataclass(frozen=True, slots=True)
class CollectionStats:
    name: str
    provider: str
    model: str
    dimensions: int
    vectors: int | None = None
    """`None` when Chroma could not be reached — not zero, which means empty."""


@dataclass(frozen=True, slots=True)
class ServiceStats:
    corpus: CorpusStats
    tokens: TokenUsage
    collections: tuple[CollectionStats, ...] = field(default_factory=tuple)
    telemetry_dropped: int = 0
    telemetry_written: int = 0
    telemetry_queue_depth: int = 0
    audit_spill_depth: int = 0
    recent_bursts: int = 0


class StatsService:
    def __init__(
        self,
        *,
        sessions: SessionSource,
        vectors: VectorStore,
        providers: ProviderRegistry,
        telemetry: TelemetrySink,
        trail: AuditTrail,
        repository: StatsRepository | None = None,
    ) -> None:
        self._sessions = sessions
        self._vectors = vectors
        self._providers = providers
        self._telemetry = telemetry
        self._trail = trail
        self._repository = repository or StatsRepository()

    async def collect(
        self, *, token_window_days: int = 30, burst_window_days: int = 7
    ) -> ServiceStats:
        async with self._sessions.session() as session:
            corpus = await self._repository.corpus(session)
            tokens = await self._repository.token_usage(session, days=token_window_days)
            bursts = await self._repository.recent_bursts(session, days=burst_window_days)

        return ServiceStats(
            corpus=corpus,
            tokens=tokens,
            collections=await self._collections(),
            telemetry_dropped=self._telemetry.dropped,
            telemetry_written=self._telemetry.written,
            telemetry_queue_depth=self._telemetry.depth,
            audit_spill_depth=await self._trail.spill_depth(),
            recent_bursts=bursts,
        )

    async def _collections(self) -> tuple[CollectionStats, ...]:
        """One entry per configured provider, whether or not it has been used.

        A provider configured and never ingested into is a real state — AD-006
        makes populating a second collection an operational choice — and showing
        it at zero is how an operator sees the comparison has not been run yet.
        """
        collected: list[CollectionStats] = []
        for name in self._providers.names:
            target = self._providers.resolve(name)
            model = target.provider.model
            collected.append(
                CollectionStats(
                    name=target.collection,
                    provider=name,
                    model=model.model_id,
                    dimensions=model.dimensions,
                    vectors=await self._count(target.collection),
                )
            )
        return tuple(collected)

    async def _count(self, collection: str) -> int | None:
        try:
            if not await self._vectors.collection_exists(collection):
                return 0
            return await self._vectors.count(collection)
        except Exception as exc:
            # Stats during a Chroma outage must still answer. The `None` is the
            # honest reading — "not known" rather than "none" — and every other
            # figure on the response is still worth having.
            _logger.warning("stats_collection_unreadable", collection=collection, exc_info=exc)
            return None
