"""Configuration, and nothing else. The machinery is in `platform_db.migrations`.

Three metadata objects, because `kb_audit` has two owners: `audit_metadata` is
the library's tier-1 table (AD-018), `telemetry_metadata` is this service's
tier-2 tables, and `kb_metadata` is the knowledge base itself. Autogenerate
compares the union against the database, so leaving one out would make the next
revision propose dropping its tables.
"""

from kb_api.adapters.postgres import KB_SCHEMA, kb_metadata, telemetry_metadata
from kb_api.config import get_database_config
from platform_db import AUDIT_SCHEMA, audit_metadata, run_migrations

run_migrations(
    target_metadata=[kb_metadata, audit_metadata, telemetry_metadata],
    dsn=get_database_config().postgres.dsn.get_secret_value(),
    schemas=(KB_SCHEMA, AUDIT_SCHEMA),
)
