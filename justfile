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

# Run the Bruno collection against a live instance.
#
# `production` points at the tailnet address (AD-023), so it only resolves from
# a tailnet-joined machine — which is where the post-deploy smoke test was
# always going to run from.
bru env="local":
    cd apps/kb-api/bruno && bru run --env {{env}} -r

# --- operator CLI ---------------------------------------------------------
#
# `kb` is a client of the deployed service (AD-025), so the useful form is the
# installed one — these recipes are for working *on* it, not for using it.

# Run the CLI from the workspace, without installing it
kb *args:
    uv run kb {{args}}

# `uv tool install` resolves `platform-core` from the workspace by path, so the
# installed tool references this checkout: moving or deleting the repo breaks
# it, and a change to `libs/platform-core` needs this rerun. That is the trade
# for not publishing two packages to an index nobody else reads.
#
# Install `kb` onto this machine as a standalone tool
kb-install:
    uv tool install ./tools/kb-cli --force
    @echo 'installed — run "kb config init", then "kb" for the browser'

kb-uninstall:
    uv tool uninstall kb-cli

# --- containers -----------------------------------------------------------
#
# Built with Podman here, with Docker on the VPS, from one Containerfile
# (AD-022). The recipes below are the local half; `docs/DEPLOYMENT.md` is the
# host half, and every command there passes `--env-file .env.prod` explicitly.

# Build the runtime image from the workspace root
image tag="kb-api:latest":
    podman build -f apps/kb-api/Containerfile -t {{tag}} .

# Validate compose.prod.yml — YAML, interpolation, and the merged result.
#
# Reads `.env.prod` if it exists and `.env.prod.example` otherwise, so this is
# runnable on a machine that holds no production secrets. The example leaves the
# secrets blank on purpose and the file declares them `${…:?}`, so the fallback
# supplies placeholders — the check being made here is that the file parses, not
# that this machine could deploy it.
#
# podman-compose, because that is what is installed here (AD-015); the VPS runs
# `docker compose config` against the same file, which is the authority. AD-022
# permits Docker-only features in this file, so a podman-compose complaint is
# worth reading before it is worth acting on.
prod-config:
    #!/usr/bin/env bash
    set -euo pipefail
    env_file=.env.prod
    if [ ! -f "$env_file" ]; then
        env_file=.env.prod.example
        export KB_POSTGRES_PASSWORD=placeholder KB_TAILSCALE_AUTHKEY=placeholder
    fi
    podman-compose --env-file "$env_file" -f compose.prod.yml config >/dev/null
    echo "compose.prod.yml parses and interpolates (from $env_file)"

# AD-023's rule, checked against the file. `deploy/verify-exposure.sh` checks it
# against a running stack; this is the half that needs no VPS.
verify-exposure:
    uv run pytest apps/kb-api/tests/test_compose_prod.py -q

# --- operations (run these on the VPS) ------------------------------------

backup:
    ./deploy/backup.sh

# Defaults to a rehearsal against a throwaway container — the live stack is
# untouched unless `--in-place` is passed. Phase 9's exit criterion is that this
# has actually been run once: a backup is not done until a restore has been.
restore backup_dir:
    ./deploy/restore.sh {{backup_dir}}
