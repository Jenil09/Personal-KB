"""The autogenerate filters, which decide what a migration is allowed to touch.

`include_schemas=True` is required for schema-qualified metadata and is also how
autogenerate learns about every other schema in the database. Without these
filters it proposes dropping tables it was simply never told about — on a shared
instance, someone else's.
"""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Column, MetaData, Table, Text

from platform_db import run_migrations
from platform_db.migrations import _only_owned_names, _only_owned_schemas

OWNED = ("kb", "kb_audit")
DSN = "postgresql+asyncpg://kb:kb@localhost:5433/kb"


def table_in(schema: str | None) -> Table:
    return Table("documents", MetaData(schema=schema), Column("id", Text))


def test_run_migrations_needs_a_schema_to_own() -> None:
    with pytest.raises(ValueError, match="at least one schema"):
        run_migrations(target_metadata=MetaData(), dsn="postgresql+asyncpg://x/y", schemas=())


@pytest.mark.parametrize("schema", ["kb", "kb_audit"])
def test_our_own_tables_are_compared(schema: str) -> None:
    include = _only_owned_schemas(OWNED)

    assert include(table_in(schema), "documents", "table", True, None) is True


def test_another_services_tables_are_left_alone() -> None:
    include = _only_owned_schemas(OWNED)

    assert include(table_in("other_service"), "documents", "table", True, None) is False


def test_an_unqualified_object_is_compared() -> None:
    # Columns and indexes arrive with no schema of their own; excluding them
    # would empty every migration of its contents.
    include = _only_owned_schemas(OWNED)

    assert include(Column("id", Text), "id", "column", True, None) is True


def test_reflection_only_descends_into_owned_schemas() -> None:
    include = _only_owned_names(OWNED)

    assert include("kb", "schema", {}) is True
    assert include("public", "schema", {}) is False
    assert include("pg_catalog", "schema", {}) is False


def test_names_other_than_schemas_are_left_to_the_object_filter() -> None:
    include = _only_owned_names(OWNED)

    assert include("documents", "table", {"schema_name": "kb"}) is True


def test_offline_mode_emits_sql_without_a_database(tmp_path: Path, capsys) -> None:
    """`alembic upgrade --sql` is how a deploy gets reviewed before it runs.

    Phase 9 applies migrations from an init container, so the offline path is
    not a curiosity — it is the one a human reads. It must work with nothing
    listening on the DSN, which is why this needs no container.
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
        f"    dsn={DSN!r},\n"
        '    schemas=("kb", "kb_audit"),\n'
        ")\n",
        encoding="utf-8",
    )
    revision = scripts / "versions" / "0001_audit.py"
    revision.write_text(
        '"""audit trail"""\n'
        "from alembic import op\n"
        "import sqlalchemy as sa\n"
        "\n"
        'revision = "0001"\n'
        "down_revision = None\n"
        "\n"
        "def upgrade() -> None:\n"
        '    op.create_table("marker", sa.Column("id", sa.Integer, primary_key=True),'
        ' schema="kb_audit")\n'
        "\n"
        "def downgrade() -> None:\n"
        '    op.drop_table("marker", schema="kb_audit")\n',
        encoding="utf-8",
    )

    command.upgrade(config, "base:head", sql=True)

    emitted = capsys.readouterr().out
    assert "CREATE TABLE kb_audit.marker" in emitted
    # The version table goes in the service's own schema, not `public`.
    assert "kb.alembic_version" in emitted
