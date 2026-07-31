"""Shared Alembic wiring, so a service's `env.py` is configuration and nothing else.

Each service keeps its own `alembic.ini` and `migrations/versions/` — revisions
are service history and do not belong in a library — but the async engine
handling, the schema creation, and the autogenerate filters are identical
everywhere and live here.

A service's `env.py` in full:

```python
from alembic import context

from kb_api.adapters.postgres.models import kb_metadata
from kb_api.config import settings
from platform_db import audit_metadata, run_migrations

run_migrations(
    target_metadata=[kb_metadata, audit_metadata],
    dsn=settings.database.dsn.get_secret_value(),
    schemas=("kb", "kb_audit"),
)
```

Two things this gets right that hand-rolled `env.py` files usually do not.
`include_schemas=True` without an `include_object` filter makes autogenerate
propose dropping every table in the database it was not told about, which on a
shared instance means someone else's. And Alembic will not create the schema its
own version table lives in, so the first `upgrade head` fails on a clean database
unless the schemas are created first.
"""

import asyncio
from collections.abc import Sequence

from alembic import context
from alembic.operations import MigrationScript
from alembic.runtime.environment import (
    IncludeNameFn,
    IncludeObjectFn,
    NameFilterParentNames,
    NameFilterType,
)
from sqlalchemy import Connection, MetaData, schema
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.schema import SchemaItem

__all__ = ["run_migrations"]


def run_migrations(
    *,
    target_metadata: MetaData | Sequence[MetaData],
    dsn: str,
    schemas: Sequence[str],
    version_table_schema: str | None = None,
) -> None:
    """Run the migration Alembic invoked us for, offline or online.

    `version_table_schema` defaults to the first entry in `schemas`, which keeps
    `alembic_version` inside the service's own namespace rather than in `public`
    where a second service would collide with it.
    """
    owned = tuple(schemas)
    if not owned:
        raise ValueError("run_migrations requires at least one schema")
    version_schema = version_table_schema or owned[0]

    if context.is_offline_mode():
        _run_offline(target_metadata, dsn, owned, version_schema)
    else:
        asyncio.run(_run_online(target_metadata, dsn, owned, version_schema))


def _run_offline(
    target_metadata: MetaData | Sequence[MetaData],
    dsn: str,
    owned: tuple[str, ...],
    version_schema: str,
) -> None:
    """Emit SQL to stdout without connecting — `alembic upgrade head --sql`.

    The deploy path runs migrations from an init container (Phase 9), and this
    is how the SQL gets reviewed before it is applied to production.
    """
    _configure(target_metadata, owned, version_schema, url=dsn)
    with context.begin_transaction():
        context.run_migrations()


async def _run_online(
    target_metadata: MetaData | Sequence[MetaData],
    dsn: str,
    owned: tuple[str, ...],
    version_schema: str,
) -> None:
    # NullPool: migrations are a one-shot process. Pooling would only leave
    # connections open past the last statement.
    engine = async_engine_from_config(
        {"sqlalchemy.url": dsn},
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_create_schemas, owned)
            await connection.commit()
            await connection.run_sync(_migrate, target_metadata, owned, version_schema)
    finally:
        await engine.dispose()


def _create_schemas(connection: Connection, owned: tuple[str, ...]) -> None:
    for name in owned:
        connection.execute(schema.CreateSchema(name, if_not_exists=True))


def _migrate(
    connection: Connection,
    target_metadata: MetaData | Sequence[MetaData],
    owned: tuple[str, ...],
    version_schema: str,
) -> None:
    _configure(target_metadata, owned, version_schema, connection=connection)
    with context.begin_transaction():
        context.run_migrations()


def _configure(
    target_metadata: MetaData | Sequence[MetaData],
    owned: tuple[str, ...],
    version_schema: str,
    *,
    connection: Connection | None = None,
    url: str | None = None,
) -> None:
    """Both modes configure identically apart from where the SQL goes.

    One call site rather than two: an option added to one and forgotten in the
    other is how `--sql` output stops matching what `upgrade head` actually
    applies. `literal_binds` and `dialect_opts` are inert online.
    """
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        literal_binds=url is not None,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_object=_only_owned_schemas(owned),
        include_name=_only_owned_names(owned),
        version_table_schema=version_schema,
        # Type and default drift is the kind that survives review and then
        # surprises someone in production; make autogenerate notice it.
        compare_type=True,
        compare_server_default=True,
        process_revision_directives=_skip_empty_revisions,
    )


def _only_owned_schemas(owned: tuple[str, ...]) -> IncludeObjectFn:
    """Keep autogenerate's attention on our own schemas.

    Without this, `include_schemas=True` reflects every schema in the database
    and proposes dropping every table it was not handed metadata for.
    """

    def include_object(
        obj: SchemaItem,
        name: str | None,
        type_: NameFilterType,
        reflected: bool,
        compare_to: SchemaItem | None,
    ) -> bool:
        table_schema = getattr(obj, "schema", None)
        return table_schema is None or table_schema in owned

    return include_object


def _only_owned_names(owned: tuple[str, ...]) -> IncludeNameFn:
    """Stop reflection descending into schemas we do not own in the first place."""

    def include_name(
        name: str | None,
        type_: NameFilterType,
        parent_names: NameFilterParentNames,
    ) -> bool:
        return name in owned if type_ == "schema" else True

    return include_name


def _skip_empty_revisions(
    migration_context: object,
    revision: object,
    directives: list[MigrationScript],
) -> None:
    """Drop a no-op autogenerate rather than committing an empty revision file."""
    if (
        directives
        and directives[0].upgrade_ops is not None
        and directives[0].upgrade_ops.is_empty()
    ):
        directives.clear()
