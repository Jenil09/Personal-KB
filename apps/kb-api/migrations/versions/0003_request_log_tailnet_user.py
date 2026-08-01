"""request_logs.tailnet_user

AD-023 routes operator traffic through the `tailscale` container, so `client_ip`
is that one proxy's Docker address on every operator request. The decision names
the recovery — `tailscale serve` forwards the tailnet identity in
`Tailscale-User-Login` — and says the tier-1 writer should record it. It says
Phase 8 is the place, and Phase 8 wired the middleware without wiring this; the
column is added here, alongside the compose file that creates the proxy hop it
exists to see through.

Nullable rather than defaulted. The n8n path has no tailnet identity and never
will, and an empty string there would be a value that reads as an answer.

No index. The forensic questions this column serves are asked about a window of
operator traffic, which `ix_request_logs_key_id_created_at` already answers —
there is exactly one operator key (AD-011), so filtering it by `key_id` first
leaves a handful of rows for the `tailnet_user` predicate to sort through. An
index here would cost a write on every request to serve a query that is already
fast.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from platform_db import AUDIT_SCHEMA

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_logs",
        sa.Column("tailnet_user", sa.Text(), nullable=True),
        schema=AUDIT_SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("request_logs", "tailnet_user", schema=AUDIT_SCHEMA)
