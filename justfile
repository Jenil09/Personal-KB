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

# `tools` is omitted until it holds Python; mypy errors on empty directories
typecheck:
    uv run mypy apps libs

# Everything CI runs, in CI's order
check: lock-check lint fmt-check typecheck test

# --- tests ----------------------------------------------------------------

test:
    uv run pytest -m "not integration"

# Requires `just up` first
test-int:
    uv run pytest -m integration

test-all:
    uv run pytest

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

run:
    uv run uvicorn kb_api.main:app --reload --port 8000

# Export the OpenAPI schema; CI fails if this produces a diff
schema:
    uv run python -m kb_api.scripts.export_openapi

# Run the Bruno collection against a live instance
bru env="local":
    cd apps/kb-api/bruno && bru run --env {{env}} -r
