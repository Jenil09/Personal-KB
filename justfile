# Redshift7 monorepo task runner.
# Run `just` with no arguments to list recipes.

set dotenv-load := true

# podman-compose is pinned deliberately — see ai-kb/DECISIONS.md AD-015
compose := "podman-compose -f compose.dev.yml"

default:
    @just --list

# --- workspace ------------------------------------------------------------

# Sync every workspace member, not just the root
sync:
    uv sync --all-packages

# Verify the lockfile is current without modifying it
lock-check:
    uv lock --check

# --- quality --------------------------------------------------------------

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

# `tools` is omitted until it holds Python; mypy errors on empty directories.
#
# Sources go in one pass. Each member's tests need their own, because mypy maps
# `libs/*/tests/test_settings.py` to the module `test_settings` whichever member
# it belongs to, and two members having one is normal. `no-untyped-def` is off
# there so pytest fixture parameters can stay unannotated; the rest is strict.
#
# Do not "fix" the duplicate-module error by adding `tests/__init__.py`: pytest
# then resolves both files to `tests.test_settings`, silently runs one twice and
# the other never.
typecheck:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run mypy apps libs --exclude '/tests/'
    for suite in $(find apps libs -type d -name tests | sort); do
        uv run mypy "$suite" --disable-error-code=no-untyped-def
    done

# Everything CI runs, in CI's order
check: lock-check lint fmt-check typecheck schema-check test

# --- tests ----------------------------------------------------------------

test:
    uv run pytest -m "not integration"

# Requires `just up` first
test-int:
    uv run pytest -m integration

test-all:
    uv run pytest

# Branch coverage for one package, with the floor its phase committed to
cov package="platform_core" floor="100":
    uv run pytest -m "not integration" --cov={{package}} --cov-branch --cov-report=term-missing --cov-fail-under={{floor}}

# --- local dependencies ---------------------------------------------------

up:
    {{compose}} up -d
    @echo "waiting for healthchecks..."
    @timeout 90 sh -c 'until podman healthcheck run kb-postgres >/dev/null 2>&1; do sleep 1; done' || echo "WARN postgres not healthy"
    @timeout 90 sh -c 'until podman healthcheck run kb-chroma >/dev/null 2>&1; do sleep 1; done' || echo "WARN chroma not healthy"
    @podman ps --filter name=kb-

down:
    {{compose}} down

# Destroys local data volumes
nuke:
    {{compose}} down -v

logs service="":
    {{compose}} logs -f {{service}}

# --- application ----------------------------------------------------------

migrate:
    cd apps/kb-api && uv run alembic upgrade head

# Fails if the models have drifted from the migrations
migrate-check:
    cd apps/kb-api && uv run alembic check

# Roll back one revision
migrate-down revision="-1":
    cd apps/kb-api && uv run alembic downgrade {{revision}}

run:
    uv run uvicorn kb_api.main:build_app --factory --reload --port 8000

# Export the OpenAPI schema; CI fails if this produces a diff
schema:
    uv run python -m kb_api.scripts.export_openapi

# Fails if the committed schema no longer matches the routes (AD-016)
schema-check:
    uv run python -m kb_api.scripts.export_openapi --check

# Run the Bruno collection against a live instance
bru env="local":
    cd apps/kb-api/bruno && bru run --env {{env}} -r
