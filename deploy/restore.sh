#!/usr/bin/env bash
# Restores a backup taken by `backup.sh`.
#
# Phase 9's exit criterion is a *rehearsal*, not a script — the backup is not
# done until a restore has been performed once — so this defaults to the
# rehearsal and requires an explicit flag to touch the live stack.
#
#   ./deploy/restore.sh backups/20260801T101500Z            # rehearse (default)
#   ./deploy/restore.sh backups/20260801T101500Z --in-place # overwrite live data
#
# The rehearsal restores into a throwaway Postgres container and a throwaway
# Chroma volume, verifies the schemas and row counts, and destroys both. It never
# touches the running stack, which is what makes it something you can run on a
# Tuesday afternoon rather than something you first attempt during an incident.
#
# Restoring in place stops kb-api first. The alternative — restoring underneath a
# live service — leaves the API holding connections to a database being dropped,
# and the audit trail writing rows into a snapshot that is about to be replaced.
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-compose.prod.yml}
ENV_FILE=${ENV_FILE:-.env.prod}

usage() { echo "usage: $0 <backup-dir> [--in-place]" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
src=$(realpath "$1"); shift
mode=rehearse
[[ ${1:-} == "--in-place" ]] && mode=in-place

[[ -f "$src/postgres.dump" ]] || { echo "no postgres.dump in $src" >&2; exit 1; }

compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

if [[ -f "$src.sha256" ]]; then
    echo "==> verifying checksums"
    (cd "$(dirname "$src")" && sha256sum --check --quiet "$(basename "$src").sha256")
fi

# --- rehearsal -------------------------------------------------------------

if [[ $mode == rehearse ]]; then
    scratch="kb-restore-rehearsal-$$"
    echo "==> rehearsing into a throwaway container ($scratch)"
    cleanup() {
        docker rm -f "$scratch" >/dev/null 2>&1 || true
        docker volume rm -f "$scratch-chroma" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    docker run --detach --name "$scratch" \
        --env POSTGRES_USER="$KB_POSTGRES_USER" \
        --env POSTGRES_PASSWORD="$KB_POSTGRES_PASSWORD" \
        --env POSTGRES_DB="$KB_POSTGRES_DB" \
        postgres:16-alpine >/dev/null

    printf '    waiting for postgres'
    for _ in $(seq 60); do
        docker exec "$scratch" pg_isready -U "$KB_POSTGRES_USER" -d "$KB_POSTGRES_DB" \
            >/dev/null 2>&1 && break
        printf '.'; sleep 1
    done
    echo

    docker exec -i "$scratch" pg_restore \
        --username "$KB_POSTGRES_USER" --dbname "$KB_POSTGRES_DB" --no-owner \
        < "$src/postgres.dump"

    echo "==> verifying the restored database"
    docker exec -i "$scratch" psql --username "$KB_POSTGRES_USER" --dbname "$KB_POSTGRES_DB" \
        --no-psqlrc --tuples-only --command "
            SELECT 'documents:    ' || count(*) FROM kb.documents WHERE deleted_at IS NULL
            UNION ALL SELECT 'chunks:       ' || count(*) FROM kb.chunks
            UNION ALL SELECT 'request_logs: ' || count(*) FROM kb_audit.request_logs
            UNION ALL SELECT 'alembic head: ' || version_num FROM alembic_version;"

    if [[ -f "$src/chroma-data.tar.gz" ]]; then
        echo "==> verifying the chroma archive"
        docker volume create "$scratch-chroma" >/dev/null
        docker run --rm \
            --volume "$scratch-chroma":/data \
            --volume "$src":/backup:ro \
            alpine:3 sh -c 'tar xzf /backup/chroma-data.tar.gz -C /data && ls /data | head'
    fi

    echo
    echo "Rehearsal complete. Record the date and the counts above in"
    echo "docs/DEPLOYMENT.md — an unrecorded rehearsal is one nobody can rely on."
    exit 0
fi

# --- in place --------------------------------------------------------------

echo "This overwrites the live database and vector store from $src."
read -r -p "Type the database name ($KB_POSTGRES_DB) to continue: " confirm
[[ $confirm == "$KB_POSTGRES_DB" ]] || { echo "aborted"; exit 1; }

echo "==> stopping kb-api"
"${compose[@]}" stop kb-api chroma

echo "==> restoring postgres"
"${compose[@]}" exec -T postgres \
    pg_restore --username "$KB_POSTGRES_USER" --dbname "$KB_POSTGRES_DB" \
        --clean --if-exists --no-owner < "$src/postgres.dump"

if [[ -f "$src/chroma-data.tar.gz" ]]; then
    echo "==> restoring the chroma volume"
    docker run --rm \
        --volume kb_kb-chroma-data:/data \
        --volume "$src":/backup:ro \
        alpine:3 sh -c 'rm -rf /data/* && tar xzf /backup/chroma-data.tar.gz -C /data'
fi

echo "==> starting"
"${compose[@]}" start chroma
"${compose[@]}" up -d kb-api

echo
echo "Restored. Startup reconciliation runs before the first request is served,"
echo "so a document left mid-supersede in the snapshot is repaired on this boot"
echo "(services/reconciliation.py). Check /health over the tailnet before"
echo "declaring the incident over."
