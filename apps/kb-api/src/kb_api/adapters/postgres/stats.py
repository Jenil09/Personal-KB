"""Aggregates for the admin endpoint (PRD §6.7).

Separate from the document and chunk repositories because none of these are
operations on an entity — they are questions about the corpus as a whole, and
folding them into `DocumentRepository` would give the ingestion flow's port a
`count_by_status` it never calls.

Every query here is either a plain aggregate over a table whose ceiling is ~100
documents, or windowed. The audit and telemetry tables are the ones that grow —
~73k rows a year — so the two that touch them carry a time bound rather than
scanning the history to answer a question about this week.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kb_api.adapters.postgres.tables import chunks, documents
from kb_api.adapters.postgres.telemetry_tables import token_usage_logs
from platform_db import repeat_bursts

__all__ = ["CorpusStats", "StatsRepository", "TokenUsage"]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Cumulative embedding spend, from tier-2 (AD-017).

    `estimated` is separated from `exact` rather than summed with it. Gemini
    reports no token count for embeddings at all, so folding its estimate into
    one total would produce a number that looks like a billing figure and is not.
    """

    exact_tokens: int = 0
    estimated_tokens: int = 0
    api_calls: int = 0


@dataclass(frozen=True, slots=True)
class CorpusStats:
    documents_by_status: dict[str, int] = field(default_factory=dict)
    documents_by_collection: dict[str, int] = field(default_factory=dict)
    total_chunks: int = 0
    total_tokens_stored: int = 0


class StatsRepository:
    async def corpus(self, session: AsyncSession) -> CorpusStats:
        """Counts over live documents only — a soft-deleted row is not corpus."""
        alive = documents.c.deleted_at.is_(None)

        by_status = await session.execute(
            select(documents.c.status, func.count()).where(alive).group_by(documents.c.status)
        )
        by_collection = await session.execute(
            select(documents.c.collection, func.count())
            .where(alive)
            .group_by(documents.c.collection)
        )
        # Joined to `documents` rather than counting `chunks` outright: a chunk
        # whose parent is soft-deleted and not yet purged is still on the table,
        # and reporting it would contradict the document count beside it.
        chunk_totals = await session.execute(
            select(func.count(), func.coalesce(func.sum(chunks.c.token_count), 0))
            .select_from(chunks.join(documents, chunks.c.document_id == documents.c.id))
            .where(alive)
        )
        count, tokens = chunk_totals.one()

        return CorpusStats(
            documents_by_status={row[0]: int(row[1]) for row in by_status.all()},
            documents_by_collection={row[0]: int(row[1]) for row in by_collection.all()},
            total_chunks=int(count),
            total_tokens_stored=int(tokens),
        )

    async def token_usage(self, session: AsyncSession, *, days: int = 30) -> TokenUsage:
        """Embedding tokens billed over a window, split by how they were counted."""
        since = datetime.now(UTC) - timedelta(days=days)
        result = await session.execute(
            select(
                func.coalesce(
                    func.sum(token_usage_logs.c.input_tokens).filter(
                        token_usage_logs.c.token_source == "exact"
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(token_usage_logs.c.input_tokens).filter(
                        token_usage_logs.c.token_source != "exact"
                    ),
                    0,
                ),
                func.coalesce(func.sum(token_usage_logs.c.api_calls), 0),
            ).where(token_usage_logs.c.created_at >= since)
        )
        exact, estimated, calls = result.one()
        return TokenUsage(
            exact_tokens=int(exact), estimated_tokens=int(estimated), api_calls=int(calls)
        )

    async def recent_bursts(self, session: AsyncSession, *, days: int = 7, limit: int = 20) -> int:
        """How many requests were flagged as an identical-query burst (AD-014).

        The canned forensic query from `platform-db`, not a `SELECT` written
        here: the query and the partial index it rides have to agree, and they
        only reliably do when one thing owns both.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        result = await session.execute(
            select(func.count()).select_from(repeat_bursts(since=since, limit=limit).subquery())
        )
        return int(result.scalar_one())
