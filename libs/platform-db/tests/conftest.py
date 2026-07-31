"""A session that succeeds, fails, or counts, on demand.

The tier-1 guarantee is a claim about what happens when Postgres is gone. A test
that can only reach that state by killing a container tests it once, slowly, and
not at all in CI's unit job — so the failure is injected at the session here, and
`test_audit_integration.py` confirms the same behaviour against a real database
that really stops.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from platform_db import AuditRecord, AuditSettings, Outcome, SpillFile


class RecordingSession:
    """Stands in for `AsyncSession`, remembering the parameters it was given."""

    def __init__(self) -> None:
        # `Any` because these are whatever `insert()` was handed — audit rows in
        # one test, arbitrary telemetry shapes in another.
        self.executed: list[list[dict[str, Any]]] = []

    async def execute(self, statement: object, parameters: object = None) -> None:
        self.executed.append(list(parameters) if isinstance(parameters, list) else [])

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [row for batch in self.executed for row in batch]


class FakeDatabase:
    """A `SessionSource` whose availability the test controls.

    `session()` is annotated as yielding `Any` so this satisfies the protocol
    without pretending to be a real `AsyncSession`; the audit tiers only ever
    call `execute` on it.
    """

    def __init__(self) -> None:
        self.available = True
        self.session_obj = RecordingSession()
        self.commits = 0
        self.rollbacks = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        if not self.available:
            raise OperationalError("SELECT 1", {}, ConnectionRefusedError("postgres is down"))
        try:
            yield self.session_obj
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.session_obj.rows


@pytest.fixture
def database() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
def spill(tmp_path) -> SpillFile:
    return SpillFile(tmp_path / "audit" / "spill.jsonl")


@pytest.fixture
def audit_settings() -> AuditSettings:
    # Small and fast: the batching behaviour is under test, not the production
    # numbers. The saturation test overrides the queue size it needs.
    return AuditSettings(
        telemetry_queue_size=100,
        telemetry_batch_size=10,
        telemetry_flush_interval_seconds=0.02,
        telemetry_drain_timeout_seconds=2.0,
    )


@pytest.fixture
def make_record():
    """A minimal valid tier-1 record, with any field overridable."""

    def factory(**overrides: Any) -> AuditRecord:
        values: dict[str, Any] = {
            "request_id": "0f9d3e9c-6d2e-4a1c-9f1a-2b7c8d4e5f60",
            "method": "POST",
            "path": "/v1/search",
            "status_code": 200,
            "outcome": Outcome.SUCCESS,
            "latency_ms": 42,
            "key_id": "n8n",
        }
        values.update(overrides)
        return AuditRecord(**values)

    return factory


@pytest.fixture
def make_records(make_record):
    """`count` distinguishable records — `latency_ms` carries the index."""

    def factory(count: int) -> list[AuditRecord]:
        return [make_record(latency_ms=index) for index in range(count)]

    return factory
