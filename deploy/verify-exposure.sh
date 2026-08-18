#!/usr/bin/env bash
# AD-023's exposure decision, checked rather than asserted.
#
# "No container in this stack publishes a host port" is only worth what it is
# tested at, and it is the kind of property that breaks silently — one `ports:`
# line, added for a debugging session and never removed, and the service is on
# the public internet with the host firewall none the wiser.
#
# Checks, run against the *running* stack on the VPS:
#
#   1. `compose ps` reports no published port for any service
#   2. the file itself has no `ports:` key
#   3. nothing in this stack is listening on a host interface
#   4. the API (and kb-mcp) are unreachable from the host itself
#   5. from the n8n container, kb-api and kb-mcp answer; Postgres and Chroma do not
#
# Check 5 is the one that proves the network asymmetry rather than just the
# absence of a binding: n8n reaching Postgres would mean `kb-shared` and
# `kb-internal` had been collapsed into one, which no other check would notice.
#
# Exit status is the number of failed checks, so this is usable as a deploy gate.
set -uo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-compose.prod.yml}
ENV_FILE=${ENV_FILE:-.env.prod}
N8N_CONTAINER=${N8N_CONTAINER:-n8n}
compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

failures=0

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; failures=$((failures + 1)); }
note() { printf '  \033[33mSKIP\033[0m %s\n' "$1"; }

echo "1. compose reports no published ports"
published=$("${compose[@]}" ps --format '{{.Service}} {{.Publishers}}' 2>/dev/null \
    | grep -vE '\[\]$|^\S+ $' || true)
if [[ -z "$published" ]]; then
    pass "no service publishes a port"
else
    fail "published ports found:"
    printf '       %s\n' "$published"
fi

echo "2. the file itself has no ports: key"
if grep -nE '^\s*ports:' "$COMPOSE_FILE"; then
    fail "$COMPOSE_FILE declares ports:"
else
    pass "$COMPOSE_FILE has no ports: key"
fi

echo "3. nothing from this stack listens on a host interface"
# Docker's DNAT rules are installed by dockerd, so a published port shows up as
# a docker-proxy listener rather than as one owned by the container.
listeners=$(ss -tlnp 2>/dev/null | grep -E ':(8000|5432|8001|5433|9000)\b' || true)
if [[ -z "$listeners" ]]; then
    pass "no listener on 8000/5432/8001/5433/9000"
else
    fail "host listeners found:"
    printf '       %s\n' "$listeners"
fi

echo "4. the API and kb-mcp are unreachable from the host"
if curl --silent --show-error --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    fail "http://127.0.0.1:8000/health answered from the host"
else
    pass "http://127.0.0.1:8000/health does not answer from the host"
fi
if curl --silent --show-error --max-time 3 http://127.0.0.1:9000/health >/dev/null 2>&1; then
    fail "http://127.0.0.1:9000/health answered from the host"
else
    pass "http://127.0.0.1:9000/health does not answer from the host"
fi

echo "5. from n8n: kb-api and kb-mcp yes, datastores no"
if ! docker ps --format '{{.Names}}' | grep -qx "$N8N_CONTAINER"; then
    note "container '$N8N_CONTAINER' not running; set N8N_CONTAINER to check this"
else
    # n8n ships wget rather than curl.
    probe() { docker exec "$N8N_CONTAINER" timeout 4 wget -q -O- "$1" >/dev/null 2>&1; }
    tcp() { docker exec "$N8N_CONTAINER" timeout 4 sh -c "nc -z $1 $2" >/dev/null 2>&1; }

    if probe http://kb-api:8000/health; then
        pass "n8n reaches kb-api on kb-shared"
    else
        fail "n8n cannot reach http://kb-api:8000/health — is it on kb-shared?"
    fi
    if probe http://kb-mcp:9000/health; then
        pass "n8n reaches kb-mcp on kb-shared"
    else
        fail "n8n cannot reach http://kb-mcp:9000/health — is it on kb-shared?"
    fi
    if tcp postgres 5432; then
        fail "n8n reached Postgres; kb-internal is not isolating it"
    else
        pass "n8n cannot reach Postgres"
    fi
    if tcp chroma 8000; then
        fail "n8n reached Chroma; kb-internal is not isolating it"
    else
        pass "n8n cannot reach Chroma"
    fi
fi

echo
if (( failures == 0 )); then
    echo "AD-023 holds: no host port, and each consumer reaches kb-api / kb-mcp and nothing else."
else
    echo "$failures check(s) failed."
fi
exit "$failures"
