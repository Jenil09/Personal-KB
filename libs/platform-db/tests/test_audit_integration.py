"""The claims only a real Postgres can settle.

Two of them, both stated as Phase 3 exit criteria:

- The tier-1 writer loses **zero** records when Postgres is killed mid-run. Not
  a stubbed session raising `OperationalError` — a database that actually stops
  answering, with connections already open in the pool.
- The Alembic scaffolding produces a migration that applies to a clean database,
  creating both schemas and keeping `alembic_version` out of `public`.

Postgres is bound to a fixed host port so the recovered container comes back at
the address the engine already holds. Restarting on a fresh ephemeral port would
model a reconnection to somewhere else, which is not the failure in question.
"""

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

from platform_db import (
    AuditTrail,
    Database,
    DatabaseSettings,
    SpillFile,
    audit_metadata,
    request_logs,
)

pytestmark = pytest.mark.integration

IMAGE = "docker.io/library/postgres:16-alpine"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def start_postgres(port: int) -> PostgresContainer:
    container = PostgresContainer(IMAGE).with_bind_ports(5432, port)
    container.start()
    return container


def dsn_for(port: int) -> str:
    return f"postgresql+asyncpg://test:test@127.0.0.1:{port}/test"


async def create_audit_schema(database: Database) -> None:
    async with database.engine.begin() as connection:
        await connection.execute(text("CREATE SCHEMA IF NOT EXISTS kb_audit"))
        await connection.run_sync(audit_metadata.create_all)


async def count_rows(database: Database) -> int:
    async with database.engine.connect() as connection:
        result = await connection.execute(select(func.count()).select_from(request_logs))
        return int(result.scalar_one())


@pytest.fixture
def port() -> int:
    return free_port()


@pytest.fixture
def postgres(port: int) -> Iterator[PostgresContainer]:
    container = start_postgres(port)
    try:
        yield container
    finally:
        # Fixtures own teardown: Ryuk is disabled under rootless Podman
        # (AD-015). Suppressed because tests here stop the container themselves.
        with suppress(Exception):
            container.stop()


@pytest.fixture
async def database(postgres: PostgresContainer, port: int) -> AsyncIterator[Database]:
    instance = Database(DatabaseSettings(dsn=dsn_for(port), connect_timeout_seconds=2.0))
    await create_audit_schema(instance)
    try:
        yield instance
    finally:
        await instance.dispose()


async def test_a_healthy_write_lands_in_postgres(
    database: Database, tmp_path: Path, make_record
) -> None:
    trail = AuditTrail(database, SpillFile(tmp_path / "spill.jsonl"))

    await trail.record(make_record())

    assert await count_rows(database) == 1
    assert await trail.spill_depth() == 0


async def test_killing_postgres_mid_run_loses_no_records(
    postgres: PostgresContainer,
    database: Database,
    port: int,
    tmp_path: Path,
    make_records,
) -> None:
    """The guarantee end to end: kill it, keep serving, drain on recovery."""
    trail = AuditTrail(database, SpillFile(tmp_path / "spill.jsonl"))
    for record in make_records(5):
        await trail.record(record)
    assert await count_rows(database) == 5

    postgres.stop()

    for record in make_records(7):
        # No exception here is half the guarantee: the request survives the
        # outage even though its audit row cannot be written.
        await trail.record(record)
    assert await trail.spill_depth() == 7

    recovered = start_postgres(port)
    try:
        await create_audit_schema(database)

        assert await trail.reconcile() == 7

        assert await count_rows(database) == 7
        assert await trail.spill_depth() == 0
    finally:
        recovered.stop()


async def test_spilled_records_keep_the_time_of_the_request(
    postgres: PostgresContainer,
    database: Database,
    port: int,
    tmp_path: Path,
    make_record,
) -> None:
    trail = AuditTrail(database, SpillFile(tmp_path / "spill.jsonl"))
    postgres.stop()

    original = make_record()
    await trail.record(original)

    recovered = start_postgres(port)
    try:
        await create_audit_schema(database)
        await trail.reconcile()

        async with database.engine.connect() as connection:
            stored = await connection.execute(select(request_logs.c.created_at))
        # The reconciler does not get to relabel when the request happened.
        assert stored.scalar_one() == original.created_at
    finally:
        recovered.stop()


async def test_the_session_scope_rolls_back_on_error(database: Database, make_record) -> None:
    with pytest.raises(RuntimeError):
        async with database.session() as session:
            await session.execute(request_logs.insert(), [make_record().to_row()])
            raise RuntimeError("the handler failed after writing")

    assert await count_rows(database) == 0


async def test_the_session_scope_commits_on_success(database: Database, make_record) -> None:
    async with database.session() as session:
        await session.execute(request_logs.insert(), [make_record().to_row()])

    assert await count_rows(database) == 1


async def test_ping_answers_true_against_a_live_database(database: Database) -> None:
    assert await database.ping() is True


async def test_the_request_dependency_commits_what_a_handler_wrote(
    database: Database, make_record
) -> None:
    # `session_dependency` is what routers actually bind. FastAPI drives it as
    # a generator, so the commit happens after the handler returns, not inside
    # it — which is the part a direct `session()` test does not exercise.
    generator = database.session_dependency()
    session = await anext(generator)
    await session.execute(request_logs.insert(), [make_record().to_row()])
    with pytest.raises(StopAsyncIteration):
        await anext(generator)

    assert await count_rows(database) == 1


async def test_the_request_dependency_rolls_back_a_failed_handler(
    database: Database, make_record
) -> None:
    generator = database.session_dependency()
    session = await anext(generator)
    await session.execute(request_logs.insert(), [make_record().to_row()])

    # FastAPI throws the handler's exception back in at the yield; that is what
    # turns a failed request into a rollback rather than a partial write.
    with pytest.raises(RuntimeError):
        await generator.athrow(RuntimeError("the handler raised"))

    assert await count_rows(database) == 0


def test_the_alembic_scaffolding_migrates_a_clean_database(
    postgres: PostgresContainer, port: int, tmp_path: Path
) -> None:
    """A service's `env.py` is the ten lines the module docstring promises.

    Autogenerate against `audit_metadata` has to produce a revision that applies
    to an empty database — which means the scaffolding created both schemas
    first, since Alembic will not create the one its own version table lives in.
    """
    scripts = tmp_path / "migrations"
    config = Config(str(tmp_path / "alembic.ini"))
    config.set_main_option("script_location", str(scripts))
    command.init(config, str(scripts))
    (scripts / "env.py").write_text(
        "from platform_db import audit_metadata, run_migrations\n"
        "\n"
        "run_migrations(\n"
        "    target_metadata=audit_metadata,\n"
        f"    dsn={dsn_for(port)!r},\n"
        '    schemas=("kb", "kb_audit"),\n'
        ")\n",
        encoding="utf-8",
    )

    command.revision(config, message="audit trail", autogenerate=True)
    command.upgrade(config, "head")

    # Verified in its own loop: the Alembic commands above are synchronous and
    # start one of their own, so this test cannot be async.
    schemas, placed = asyncio.run(_inspect(port))

    assert {"kb", "kb_audit"} <= schemas
    assert placed["request_logs"] == "kb_audit"
    # `alembic_version` in `public` is how two services sharing an instance end
    # up overwriting each other's migration state.
    assert placed["alembic_version"] == "kb"


async def _inspect(port: int) -> tuple[set[str], dict[str, str]]:
    engine = create_async_engine(dsn_for(port))
    try:
        async with engine.connect() as connection:
            schemas = set(
                (await connection.execute(text("SELECT nspname FROM pg_namespace"))).scalars()
            )
            rows = await connection.execute(
                text("SELECT table_name, table_schema FROM information_schema.tables")
            )
            placed = {str(name): str(table_schema) for name, table_schema in rows.all()}
    finally:
        await engine.dispose()
    return schemas, placed
