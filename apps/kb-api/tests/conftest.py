"""A containerised Postgres with the real migrations applied.

The schema under test is the one `alembic upgrade head` produces, not one
`metadata.create_all` invents. They are supposed to be the same thing, and the
exit criterion is that nobody takes that on trust — a `create_all` fixture would
test the models against themselves and never touch the migration at all.

One container per module, since starting Postgres costs more than every test in
a file put together. Tests do not inherit each other's rows: `database` truncates
on the way out.

Everything shared is a fixture rather than an importable helper. `tests/` has no
`__init__.py` (deliberately — see CLAUDE.md), and under `--import-mode=importlib`
that makes `from tests.conftest import ...` fail.
"""

import hashlib
import os
import socket
import subprocess
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

from kb_api.domain import NewChunk, NewDocument
from platform_db import Database, DatabaseSettings

IMAGE = "docker.io/library/postgres:16-alpine"

KB_API_ROOT = Path(__file__).resolve().parents[1]

Alembic = Callable[..., subprocess.CompletedProcess[str]]

# The v1 collection name (Design §2.3). Tests that care about collection
# scoping build a second one rather than reusing this.
COLLECTION = "kb__openai__text_embedding_3_small__1536__c1"

MODEL_ID = "text-embedding-3-small"

_NAMESPACE = UUID("6f9e2a7c-8b31-4f0d-9a12-6c5d4e3b2a10")

TRUNCATE = text(
    "TRUNCATE kb.documents, kb.chunks, kb_audit.request_logs, "
    "kb_audit.token_usage_logs, kb_audit.ingest_logs, kb_audit.error_logs "
    "RESTART IDENTITY CASCADE"
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def postgres_port() -> int:
    return _free_port()


@pytest.fixture(scope="module")
def dsn(postgres_port: int) -> str:
    return f"postgresql+asyncpg://test:test@127.0.0.1:{postgres_port}/test"


@pytest.fixture(scope="module")
def postgres(postgres_port: int) -> Iterator[PostgresContainer]:
    container = PostgresContainer(IMAGE).with_bind_ports(5432, postgres_port)
    container.start()
    try:
        yield container
    finally:
        # Fixtures own teardown: Ryuk is disabled under rootless Podman (AD-015).
        with suppress(Exception):
            container.stop()


@pytest.fixture(scope="module")
def alembic(postgres: PostgresContainer, dsn: str) -> Alembic:
    """Run an alembic command against the container.

    A subprocess rather than `alembic.command`, because that is how it runs in
    CI and on deploy — and because the Alembic API is synchronous and starts an
    event loop of its own, which cannot be done from inside a running async test.
    """

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["uv", "run", "alembic", *args],
            cwd=KB_API_ROOT,
            env={**os.environ, "KB_API__POSTGRES__DSN": dsn},
            capture_output=True,
            text=True,
            check=True,
        )

    return run


@pytest.fixture(scope="module")
def migrated(alembic: Alembic, dsn: str) -> str:
    alembic("upgrade", "head")
    return dsn


@pytest.fixture
async def database(migrated: str) -> AsyncIterator[Database]:
    instance = Database(DatabaseSettings(dsn=migrated, connect_timeout_seconds=5.0))
    try:
        yield instance
        async with instance.engine.begin() as connection:
            await connection.execute(TRUNCATE)
    finally:
        await instance.dispose()


# --- test data ------------------------------------------------------------
#
# Fixtures rather than an importable `factories.py`: `tests/` has no
# `__init__.py` and importlib mode keeps it off `sys.path`, so a sibling module
# is not importable from a test. Matches `make_record` in platform-db's suite.


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _stable_id(*parts: str) -> UUID:
    """A UUID derived from its inputs, so a rerun produces the same one."""
    return uuid5(_NAMESPACE, "/".join(parts))


@pytest.fixture
def make_document():
    """A valid `NewDocument`, with a correctly derived `content_hash`.

    The hash is computed from the content rather than passed in, because a
    fixture that let them disagree would make the AD-008 idempotency tests pass
    against data the real flow cannot produce.
    """

    def factory(
        name: str = "redshift-architecture",
        *,
        content: str | None = None,
        collection: str = COLLECTION,
        tags: tuple[str, ...] = (),
        **overrides: Any,
    ) -> NewDocument:
        body = content if content is not None else f"# {name}\n\nContent of {name}."
        values: dict[str, Any] = {
            "id": _stable_id("document", name, collection),
            "title": name.replace("-", " ").title(),
            "content": body,
            "content_hash": _sha256(body),
            "type": "architecture",
            "collection": collection,
            "tags": tags,
        }
        values.update(overrides)
        return NewDocument(**values)

    return factory


@pytest.fixture
def make_chunk():
    """A `NewChunk` whose `text_hash` is `sha256(text + model_id)` (AD-008).

    Derived here for the same reason: the carry-forward tests are worthless if
    the fixture hashes something other than what the ingestion flow will.
    """

    def factory(
        document_id: UUID,
        ordinal: int,
        text_value: str,
        *,
        model_id: str = MODEL_ID,
        token_count: int = 128,
    ) -> NewChunk:
        return NewChunk(
            id=_stable_id("chunk", str(document_id), str(ordinal)),
            document_id=document_id,
            ordinal=ordinal,
            text=text_value,
            text_hash=_sha256(text_value + model_id),
            token_count=token_count,
            chroma_id=f"{document_id}:{ordinal}",
        )

    return factory
