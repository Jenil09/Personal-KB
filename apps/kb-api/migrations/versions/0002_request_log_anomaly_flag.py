"""request_logs.anomaly

AD-014's middle threshold — 600 requests/day for one key — flags rather than
rejects. That makes this column the only place the crossing is recorded: a
request past the threshold is served normally and is, in every other column,
indistinguishable from one below it.

Added rather than folded into 0001 because 0001 has already been applied. The
column is `NOT NULL DEFAULT false`, so the backfill is Postgres's own and every
row written before this revision correctly reads as un-flagged.

The partial index is spelled as a bare column, not `anomaly IS TRUE`. Postgres
has to prove a query's predicate implies the index's and does not make the step
from `x IS TRUE` to `x`, so the other spelling would leave the index permanently
unused — the same 76x mistake Phase 4 measured on `repeat_burst`.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from platform_db import AUDIT_SCHEMA

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_logs",
        sa.Column("anomaly", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        schema=AUDIT_SCHEMA,
    )
    op.create_index(
        "ix_request_logs_anomaly",
        "request_logs",
        ["anomaly"],
        unique=False,
        schema=AUDIT_SCHEMA,
        postgresql_where=sa.text("anomaly"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_request_logs_anomaly",
        table_name="request_logs",
        schema=AUDIT_SCHEMA,
        postgresql_where=sa.text("anomaly"),
    )
    op.drop_column("request_logs", "anomaly", schema=AUDIT_SCHEMA)
