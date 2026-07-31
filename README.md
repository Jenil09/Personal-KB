# Redshift7 — Personal KB

A **uv-workspace monorepo** for self-hosted Python services.

The first service is **`kb-api`**: a FastAPI + ChromaDB personal knowledge base providing
semantic search over personal technical documentation, consumed by n8n workflows and a local
CLI. PostgreSQL is the system of record and holds a guaranteed audit trail.

> **Status:** Early development. The workspace, tooling, and CI are in place. All four shared
> libraries are built and tested — `platform-core` (settings, logging, error taxonomy,
> correlation IDs), `platform-fastapi` (app factory, bearer auth, problem+json errors,
> health), `platform-db` (async engine, Alembic scaffolding, two-tier audit trail), and
> `ai-embeddings` (provider port, OpenAI and Gemini drivers, token accounting). `kb-api` is
> still a skeleton.

## Prerequisites

| Tool | Notes |
| --- | --- |
| [uv](https://docs.astral.sh/uv/) | Manages Python and all dependencies |
| [just](https://just.systems/) | Task runner |
| Podman + `podman-compose` | Rootless. Docker is not used — see gotchas below |

Python 3.13 is installed automatically by uv from `.python-version`.

## Quickstart

```bash
uv sync --all-packages     # or: just sync
cp .env.example .env       # then fill in provider API keys
just up                    # start Postgres + Chroma
just check                 # lockfile, lint, format, types, unit tests
```

`just up` waits for both containers to report healthy and prints their status.

## Commands

| Command | Purpose |
| --- | --- |
| `just sync` | Sync every workspace member |
| `just check` | Everything CI runs, in CI's order |
| `just lint` / `just fmt` | ruff check / ruff format |
| `just typecheck` | mypy (strict) |
| `just test` | Unit tests |
| `just test-int` | Integration tests — start their own containers via testcontainers |
| `just cov [package] [floor]` | Branch coverage for one package, floor enforced |
| `just up` / `just down` | Local Postgres + Chroma |
| `just nuke` | Tear down **and destroy data volumes** |
| `just logs [service]` | Follow container logs |
| `just migrate` | `alembic upgrade head` |
| `just run` | Run kb-api locally on :8000 |
| `just bru [env]` | Run the Bruno collection (`local` or `production`) |

Run `just` with no arguments to list everything.

## Layout

```
apps/<service>/     deployable services      (src layout, own pyproject + Containerfile)
libs/<package>/     shared platform code
tools/<tool>/       local tooling
docs/               hand-written docs
```

Workspace members declare cross-dependencies with
`[tool.uv.sources] <pkg> = { workspace = true }`. There is one root `uv.lock` and one `.venv`.

**Monorepo rule:** before adding code to `apps/`, ask whether a second service would want it —
if yes, it belongs in `libs/`. Don't extract a library speculatively; wait for a real second
consumer. Services import from `libs/`, never from each other.

## Local ports

| Service | Port | Note |
| --- | --- | --- |
| kb-api | 8000 | `just run` |
| Chroma | 8001 | mapped from container 8000 |
| PostgreSQL | 5433 | mapped from 5432, avoiding a local instance |

## Conventions

- **Feature branches always.** Branch from `master`; never commit to it directly.
  `<type>/<short-kebab-description>`, e.g. `feat/chroma-adapter`.
- **Conventional commits, title only.** One line, no body, no trailers.
  `feat(kb-api): add semantic search endpoint`. Scope is the package name, or `repo` for
  workspace-wide changes.
- **Layering:** `api → services → domain ← adapters`. Routers never touch adapters directly;
  only the composition root binds concrete implementations.
- mypy strict, ruff-formatted, async throughout.

## Gotchas

- **Use `podman-compose`, never bare `podman compose`.** On WSL2 the latter delegates to
  Docker Desktop's `docker-compose.exe` across the Windows boundary, silently changing runtime
  and networking. Verify with `podman ps`, not `docker ps`.
- **`uv sync` alone only syncs the root project.** Workspace members need `--all-packages`.
- **Testcontainers** is pointed at the rootless Podman socket by the root `conftest.py`, with
  the Ryuk reaper disabled because it is unreliable rootless. Fixtures own their teardown.
- **Chroma metadata values must be scalars** — lists are rejected, so tag filtering resolves
  through PostgreSQL.
- **The Chroma image ships no curl, wget, or python.** Its healthcheck uses bash `/dev/tcp`.
