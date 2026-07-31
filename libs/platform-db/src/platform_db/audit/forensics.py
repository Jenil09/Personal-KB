"""The questions the tier-1 trail exists to answer.

`request_logs` is not a table anyone browses. It is written to on every request
and read from perhaps twice a year, both times because something is wrong and
someone needs to know what a key did, what came from an address, what failed, or
what looped. Those four questions are exactly the four indexes in `tables.py`.

Canned here, next to the table, for the same reason AD-018 put the table in this
library rather than in the service: the query and the index it depends on have to
agree, and they only reliably do when one thing owns both. A service writing its
own `SELECT` against these columns finds out about a changed index in production.

Each function returns a `Select` rather than executing one. The caller supplies
the session — an admin endpoint's, the CLI's, or a `psql` transcript via
`str(statement.compile(...))` — and the boundary stays where the rest of the
library keeps it.

Every query here is bounded: a window, a limit, or both. An unbounded scan of the
audit table during an incident is how the investigation becomes the second
outage.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, Select, func, select

from platform_db.audit.records import Outcome
from platform_db.audit.tables import request_logs

__all__ = [
    "activity_by_ip",
    "activity_by_key",
    "failures_in_window",
    "ingests_by_key",
    "repeat_bursts",
    "traffic_summary",
]

DEFAULT_LIMIT = 500


def activity_by_key(
    key_id: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Select[Any]:
    """Everything one API key did, newest first.

    The first question after a key is suspected of being compromised or of
    driving a runaway workflow. Uses `ix_request_logs_key_id_created_at`, whose
    column order — key then time descending — is this query's shape exactly.
    """
    return (
        select(request_logs)
        .where(request_logs.c.key_id == key_id, *_window(since, until))
        .order_by(request_logs.c.created_at.desc())
        .limit(limit)
    )


def activity_by_ip(
    client_ip: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Select[Any]:
    """Everything one address did, newest first, across every key.

    Deliberately not scoped to a key: the point is to catch one source using
    several, which is what `key_id`-scoped queries cannot see. Uses
    `ix_request_logs_client_ip_created_at`.
    """
    return (
        select(request_logs)
        .where(request_logs.c.client_ip == client_ip, *_window(since, until))
        .order_by(request_logs.c.created_at.desc())
        .limit(limit)
    )


def failures_in_window(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    outcomes: tuple[Outcome, ...] = (),
    limit: int = DEFAULT_LIMIT,
) -> Select[Any]:
    """Everything that did not succeed in a window.

    `outcome <> 'success'` rather than an `IN` list by default, so a new
    `Outcome` added later shows up here without anyone remembering to add it —
    the failure you have not thought of is the one worth surfacing. Uses the
    partial `ix_request_logs_outcome`, which indexes only the non-success rows
    and is therefore a fraction of the table.

    Passing `outcomes` narrows it — auth failures alone, say — and still rides
    the same partial index as long as `success` is not among them.
    """
    condition = (
        request_logs.c.outcome.in_([outcome.value for outcome in outcomes])
        if outcomes
        else request_logs.c.outcome != Outcome.SUCCESS.value
    )
    return (
        select(request_logs)
        .where(condition, *_window(since, until))
        .order_by(request_logs.c.created_at.desc())
        .limit(limit)
    )


def repeat_bursts(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Select[Any]:
    """Requests flagged as an identical-query burst (AD-014).

    The sharpest of the four signals: a rate limit says traffic was heavy, this
    says it was *the same request over and over*, which is broken automation
    rather than load. Uses the partial `ix_request_logs_repeat_burst`, which
    holds only the flagged rows — in normal operation, none.

    The predicate is the bare column, not `.is_(True)`. Postgres has to prove a
    query's predicate implies the partial index's, and it does not make that
    step from `repeat_burst IS TRUE` to `repeat_burst`, so the spelling decides
    whether the index is considered at all: measured at 40k rows, an index scan
    costing 12 against a sequential scan costing 934.
    """
    return (
        select(request_logs)
        .where(request_logs.c.repeat_burst, *_window(since, until))
        .order_by(request_logs.c.created_at.desc())
        .limit(limit)
    )


def ingests_by_key(
    key_id: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Select[Any]:
    """Everything a key ever put into the knowledge base.

    AD-014's blast-radius query. A document found to carry a malicious payload
    is traced to its submitter through `kb.documents.ingested_by_key_id`; this
    is the other direction — given the submitter, what else came from them,
    including the attempts that failed before a row was ever written.

    Returns the trail's own record rather than joining `kb.documents`: this
    library does not know that schema exists, and the audit payload carries the
    title, source, and content hash of each ingest for exactly this reason.
    """
    return (
        select(request_logs)
        .where(
            request_logs.c.key_id == key_id,
            request_logs.c.operation == "ingest",
            *_window(since, until),
        )
        .order_by(request_logs.c.created_at.desc())
        .limit(limit)
    )


def traffic_summary(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Select[Any]:
    """Request counts and latency per key and outcome over a window.

    The one aggregate: what "normal" looks like, so AD-014's thresholds can be
    tuned against observed traffic instead of the first guesses they currently
    are. Scans the window via `ix_request_logs_created_at` rather than a
    selective predicate, which is why it is the one query here without a row
    limit — the group-by bounds the result instead.
    """
    return (
        select(
            request_logs.c.key_id,
            request_logs.c.outcome,
            func.count().label("requests"),
            func.avg(request_logs.c.latency_ms).label("avg_latency_ms"),
            func.max(request_logs.c.latency_ms).label("max_latency_ms"),
            func.min(request_logs.c.created_at).label("first_seen"),
            func.max(request_logs.c.created_at).label("last_seen"),
        )
        .where(*_window(since, until))
        .group_by(request_logs.c.key_id, request_logs.c.outcome)
        .order_by(func.count().desc())
    )


def _window(since: datetime | None, until: datetime | None) -> list[ColumnElement[bool]]:
    """Half-open `[since, until)`.

    Half-open so consecutive windows tile without double-counting the row that
    lands exactly on the boundary — which, in an incident reconstructed hour by
    hour, is the row someone is counting.
    """
    conditions: list[ColumnElement[bool]] = []
    if since is not None:
        conditions.append(request_logs.c.created_at >= since)
    if until is not None:
        conditions.append(request_logs.c.created_at < until)
    return conditions
