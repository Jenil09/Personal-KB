#!/usr/bin/env bash
# Backs up both stores, consistently enough to restore from.
#
# Two artefacts, because the two stores fail differently:
#
#   - Postgres: `pg_dump -Fc`, taken through the running container. This is the
#     system of record (AD-003) and the only one whose loss is unrecoverable.
#   - Chroma: a tar of the volume, taken with the container *stopped*. Chroma is
#     a rebuildable index (AD-004), so this is a restore-time optimisation rather
#     than a durability requirement — a corrupt snapshot costs a re-embed
#     (~$0.07 for the whole corpus), not data.
#
# The Chroma stop is why the two are not a single atomic snapshot, and why that
# is acceptable: if the dump and the tar disagree, Postgres wins and the index is
# rebuilt from it. Any other ordering would need the API stopped for the whole
# backup.
#
# The audit schema holds full search query text (AD-013), so these files carry
# the same sensitivity as the knowledge base itself: 0600, and the same access
# control wherever they are copied to.
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-compose.prod.yml}
ENV_FILE=${ENV_FILE:-.env.prod}
BACKUP_DIR=${BACKUP_DIR:-./backups}
RETAIN=${RETAIN:-14}

compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")
stamp=$(date -u +%Y%m%dT%H%M%SZ)
dest="$BACKUP_DIR/$stamp"

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

umask 077
mkdir -p "$dest"

echo "==> pg_dump -> $dest/postgres.dump"
"${compose[@]}" exec -T postgres \
    pg_dump --format=custom --compress=9 \
        --username "$KB_POSTGRES_USER" "$KB_POSTGRES_DB" \
    > "$dest/postgres.dump"

echo "==> chroma volume -> $dest/chroma-data.tar.gz"
"${compose[@]}" stop chroma >/dev/null
trap '"${compose[@]}" start chroma >/dev/null || true' EXIT
# A throwaway container with the volume attached: the tar does not depend on the
# chroma image having a shell, and it runs while nothing holds the files open.
docker run --rm \
    --volume kb_kb-chroma-data:/data:ro \
    --volume "$(realpath "$dest")":/backup \
    alpine:3 tar czf /backup/chroma-data.tar.gz -C /data .
"${compose[@]}" start chroma >/dev/null
trap - EXIT

echo "==> manifest"
{
    echo "taken_at:      $stamp"
    echo "image:         ${KB_API_IMAGE:-kb-api:latest}"
    echo "git_revision:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "alembic_head:  $("${compose[@]}" run --rm migrate alembic current 2>/dev/null | tail -1)"
} > "$dest/MANIFEST"

chmod -R 600 "$dest"/*
sha256sum "$dest"/* > "$dest.sha256"

echo "==> pruning backups older than $RETAIN days"
find "$BACKUP_DIR" -maxdepth 1 -name '20*' -mtime "+$RETAIN" -exec rm -rf {} + 2>/dev/null || true

echo
echo "Backup complete: $dest"
echo "A backup is not done until a restore has been performed. See"
echo "docs/DEPLOYMENT.md — 'Restore rehearsal'."
