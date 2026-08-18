# Deploying `kb-api` and `kb-mcp`

The VPS half of Phase 9 and Phase 11. The local half is `just image`,
`just image-mcp`, `just prod-config`, and `just verify-exposure`.

Neither service is **published to the internet** (AD-023, AD-027). There is no
Traefik router, no public hostname, no certificate, and no container in the
stack binds a host port. There are exactly three ways in:

| Consumer | Path | Auth |
| --- | --- | --- |
| n8n | `http://kb-api:8000` over the `kb-shared` Docker network | `search`-scoped API key |
| operator | Tailscale `:443`, through the `tailscale` container | `search\|write` API key + tailnet identity |
| MCP host | Tailscale `:8443` → `kb-mcp` → `kb-api` | host bearer (`KB_MCP__API_KEYS`) plus the `kb-mcp` key `kb-mcp` presents to `kb-api` |

If you find yourself adding a `ports:` line to make something work, the thing you
are trying to do has a different answer. See "Reaching the API" below.

---

## Prerequisites on the host

- Docker and Docker Compose v2 (`docker compose version`). Not Podman — AD-022
  splits the runtimes: Podman for development and CI, Docker here.
- The `kb-shared` network, which n8n owns. Create it once if it does not exist:

  ```bash
  docker network create kb-shared
  ```

  Then attach n8n to it, if it is not already:

  ```bash
  docker network connect kb-shared n8n
  ```

- A Tailscale auth key — reusable and **not** ephemeral. An ephemeral key means
  the node disappears when the container restarts, and the Bruno `production`
  environment stops resolving.
- **MagicDNS and HTTPS Certificates enabled for the tailnet**, at
  [the DNS page of the admin console](https://login.tailscale.com/admin/dns).
  `deploy/tailscale-serve.json` keys its handlers on `${TS_CERT_DOMAIN}:443`
  and `${TS_CERT_DOMAIN}:8443`, and that variable is only populated once the
  tailnet can issue certificates. Without it the node joins and is routable,
  but `containerboot` refuses to apply the serve config, so both ports fall
  through to netstack's localhost forwarder.
  The symptom is a connection failure from the operator with a healthy stack
  behind it, and these two lines on the proxy:

  ```
  boot: serve proxy: ... it is not able to issue TLS certs, so this will likely not work.
  netstack: could not connect to local backend server at 127.0.0.1:443: connection refused
  ```

  Confirm with `tailscale serve status` inside the container: it should print
  the node's HTTPS URL and **two** proxy lines (`:443` → `kb-api-upstream:8000`,
  `:8443` → `kb-mcp-upstream:9000`), not `No serve config`.
- An OpenAI API key. The corpus is bound to `text-embedding-3-small` (AD-006);
  changing the model is a full re-embed into a new collection, not a config edit.

---

## First deploy

```bash
git clone <repo> /srv/kb-api && cd /srv/kb-api

cp .env.prod.example .env.prod
chmod 600 .env.prod
$EDITOR .env.prod          # every value marked CHANGE_ME, and the two blanks

docker compose --env-file .env.prod -f compose.prod.yml build
docker compose --env-file .env.prod -f compose.prod.yml up -d
```

Every compose command passes `--env-file .env.prod`. Without it, compose
interpolates `${…}` from `.env`, the Postgres credentials resolve to empty
strings, and the stack comes up with a database nobody can log into. The
`:?set in .env.prod` markers in the compose file are there to make that failure
loud rather than silent.

Startup order is enforced by the file, not by waiting:

1. `postgres` and `chroma` come up and pass their healthchecks.
2. `migrate` runs `alembic upgrade head` **once**, from the same image, and exits.
3. `kb-api` starts only on `service_completed_successfully` — a failed migration
   means the API never starts, rather than a crash loop that reruns it every few
   seconds.
4. `kb-mcp` starts only once `kb-api` is healthy, and calls it at
   `http://kb-api:8000` over `kb-shared` — the name is unambiguous there
   because the tailscale container is not on that network.
5. `tailscale` joins the tailnet and proxies `:443` to
   `http://kb-api-upstream:8000` and `:8443` to `http://kb-mcp-upstream:9000`
   — the aliases, not the service names. The proxy container's own hostname
   is `kb-api`, since that is the tailnet node name and therefore the
   operator's URL, so the bare name is ambiguous on `kb-tailnet` and
   resolves to the proxy itself.

### Verify the deploy

```bash
# Nothing is published, and each consumer reaches kb-api / kb-mcp and nothing else.
# n8n → kb-mcp:9000/health must succeed; the host → :9000 must not.
./deploy/verify-exposure.sh

# The resolved scope grants, logged at startup (AD-024). A key missing from
# KB_API__API_KEY_SCOPES is unrestricted, not unprivileged — read this rather
# than assume it. Same for KB_MCP__API_KEY_SCOPES on the MCP process.
docker compose --env-file .env.prod -f compose.prod.yml logs kb-api | grep -i scope
docker compose --env-file .env.prod -f compose.prod.yml logs kb-mcp | grep -E 'scope|dns_rebinding'

# From a tailnet-joined machine:
curl https://kb-api.<your-tailnet>.ts.net/health
curl https://kb-api.<your-tailnet>.ts.net:8443/health
cd apps/kb-api/bruno && bru run --env production -r
```

`/health` has three states, not two. `degraded` returns **200** deliberately: a
non-empty tier-1 audit spill means Postgres was unreachable when those records
were written and the reconciler has not drained them yet. The request path is
fine; the durability of the trail is not. That wants an operator, not a restart,
and a 503 there would take a healthy instance out of rotation over a backlog that
is already retrying every thirty seconds.

---

## Reaching the API

**As n8n.** By container name on the shared network: `http://kb-api:8000/v1/search`.
Plain HTTP, never traverses the edge. The credential is the `search`-scoped key.

**As the operator.** Over the tailnet, at the `tailscale` container's node name,
port 443. Requests arrive through a proxy hop, so the tier-1 `client_ip` is that
container's Docker address for every operator request — one constant value.
`tailscale serve` forwards the real identity in `Tailscale-User-Login`, which is
recorded in `request_logs.tailnet_user` (migration `0003`). That column is how
you tell operator requests apart in the trail; `client_ip` will not do it.

**As an MCP host.** Over the tailnet, same node, port 8443 (AD-027). Two
credentials, never one string: the bearer the host presents is
`KB_MCP__API_KEYS`; the key `kb-mcp` presents to `kb-api` is the `kb-mcp`
entry in `KB_API__API_KEYS` and must match `KB_MCP__KB_API__API_KEY`. Tool
calls land in `kb_audit.request_logs` attributed to `key_id=kb-mcp` with
`User-Agent` starting `kb-mcp/`. See "Connecting an MCP host" below.

**For debugging.** `docker compose exec kb-api …`, or a throwaway container on
the network:

```bash
docker run --rm --network kb-shared curlimages/curl \
    -H "Authorization: Bearer $KEY" http://kb-api:8000/health
```

Not a published port. Not even temporarily — Docker installs its DNAT rules
ahead of the host firewall, so a `ports:` line is world-reachable no matter what
`ufw` says, and a temporary one has a way of becoming permanent.

The same throwaway-container form reaches `kb-mcp`:

```bash
docker run --rm --network kb-shared curlimages/curl \
    http://kb-mcp:9000/health
```

---

## Connecting an MCP host

From a tailnet-joined machine, Claude Code (the verification host, Stage 0):

```bash
claude mcp add --transport http kb \
    https://kb-api.<your-tailnet>.ts.net:8443/mcp \
    --header "Authorization: Bearer $KB_MCP_HOST_TOKEN"
```

`$KB_MCP_HOST_TOKEN` is the secret half of an entry in `KB_MCP__API_KEYS`,
not the `kb-mcp` key in `KB_API__API_KEYS`. A configured `Authorization`
header short-circuits RFC 9728 discovery — Claude Code never reads the
well-known route. A host that cannot set a header will follow
`authorization_servers`, find nothing, and fail; that is a real OAuth
server and is out of scope (AD-027).

Confirm both ends of a scratch ingest:

```bash
# Process is up. `/health` is liveness, not "kb-api is reachable".
curl https://kb-api.<your-tailnet>.ts.net:8443/health

# After a search and an ingest from the host, both rows belong to the
# kb-mcp key and carry its User-Agent.
docker compose --env-file .env.prod -f compose.prod.yml exec postgres \
    psql -U "$KB_POSTGRES_USER" -d "$KB_POSTGRES_DB" -c "
        SELECT created_at, key_id, method, path, status_code, user_agent
        FROM kb_audit.request_logs
        WHERE user_agent LIKE 'kb-mcp/%'
        ORDER BY created_at DESC
        LIMIT 10;"
```

---

## Updating

```bash
git pull
docker compose --env-file .env.prod -f compose.prod.yml build
docker compose --env-file .env.prod -f compose.prod.yml up -d
```

`up -d` reruns `migrate` before replacing `kb-api`, so a schema change ships with
the code that needs it. Take a backup first if the release carries a migration
that drops or rewrites anything — Alembic downgrades are written for every
revision here, but a downgrade that drops a column is not a recovery, it is a
second loss.

**Rolling back.** Set `KB_API_IMAGE` (and `KB_MCP_IMAGE`, if the release
touched `kb-mcp`) in `.env.prod` to the previous tag and `up -d`. If the
release added a migration, downgrade it explicitly first:

```bash
docker compose --env-file .env.prod -f compose.prod.yml run --rm migrate \
    alembic downgrade -1
```

---

## Backups

```bash
./deploy/backup.sh          # writes ./backups/<timestamp>/
```

Two artefacts, because the two stores fail differently. Postgres is the system of
record (AD-003) and its dump is the one whose loss is unrecoverable. Chroma is a
rebuildable index (AD-004), so its volume snapshot is a restore-time
optimisation — a corrupt one costs a re-embed of the whole corpus, about $0.07,
not data.

**These files carry the audit schema, and the audit schema holds full search
query text** (AD-013). They are as sensitive as the knowledge base itself, and
need the same access control wherever they are copied to. `backup.sh` writes them
0600; keep them that way.

Run it from cron, daily, and copy the result off the host:

```cron
15 3 * * * cd /srv/kb-api && ./deploy/backup.sh >> /var/log/kb-backup.log 2>&1
```

---

## Restore rehearsal

**A backup is not done until a restore has been performed.** Until then it is an
untested file that produces a comfortable feeling. The rehearsal is the exit
criterion, not the script.

```bash
./deploy/restore.sh backups/<timestamp>
```

The default mode restores into a throwaway Postgres container and a throwaway
Chroma volume, prints the document, chunk, and audit row counts along with the
Alembic head, and destroys both. It never touches the running stack — which is
the point: it is something to run on a Tuesday afternoon, not something to first
attempt during an incident.

Restoring for real needs `--in-place`, and prompts for the database name:

```bash
./deploy/restore.sh backups/<timestamp> --in-place
```

That stops `kb-api` and `chroma` first. Restoring underneath a live service
leaves the API holding connections to a database being dropped, and the audit
trail writing rows into a snapshot about to be replaced. On the way back up,
startup reconciliation runs before the first request is served, so a document
left mid-supersede in the snapshot is repaired on that boot rather than being
answerable by search and invisible to everything else.

### Rehearsal log

Record every rehearsal here. An unrecorded one is one nobody can rely on.

| Date | Backup | Documents | Chunks | Audit rows | Alembic head | Result |
| --- | --- | --- | --- | --- | --- | --- |
| _pending first deploy_ | | | | | | |

---

## After the first deploy: re-measure the audit write

Phase 8 measured the tier-1 synchronous audit write at **44.7 ms added to p95**
on the WSL2 development host, of which the `INSERT` itself was 1.8 ms and the
rest was commit `fsync` on a slow volume. AD-013's synchronous write is only
correct if the real number lands near its 2–5 ms estimate, and VPS NVMe fsync is
roughly two orders of magnitude faster.

```bash
docker compose --env-file .env.prod -f compose.prod.yml exec postgres \
    psql -U "$KB_POSTGRES_USER" -d "$KB_POSTGRES_DB" -c "
        SELECT percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
               percentile_disc(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
               count(*)
        FROM kb_audit.request_logs
        WHERE created_at > now() - interval '24 hours';"
```

Let a day of real n8n traffic accumulate first — a fresh deploy's numbers are
mostly cold caches. If the added cost is still tens of milliseconds on NVMe,
AD-013's synchronous write is the decision to revisit, not the index set.

---

## Troubleshooting

**`kb-api` never starts.** Check `migrate` first: `docker compose … logs migrate`.
The API is gated on that container exiting zero, so a migration failure looks
like an API that will not start.

**`no such network: kb-shared`.** It is `external: true` — n8n owns it and this
stack joins it. `docker network create kb-shared`.

**Startup fails on a provider.** A blank `KB_API__OPENAI__API_KEY=` is rejected
at boot on purpose. Leave the whole line out rather than blank; a half-configured
provider otherwise fails on the first embedding call instead of on start.

**`/health` reports `degraded`.** Read which check said so. `audit_spill: N pending`
means Postgres was unavailable and the trail spilled to disk; the reconciler
drains it automatically once Postgres returns. `telemetry_queue` over 80% or with
a non-zero drop count means tier-2 telemetry is being lost — best-effort by
contract (AD-013), so it does not affect the guaranteed trail.

**Tailscale node disappears on restart.** The auth key was ephemeral, or
`kb-tailscale-state` was removed. Both burn the node identity.

**`kb-mcp` never starts.** It waits on `kb-api` becoming healthy. Check
`docker compose … logs kb-api` first, then `logs kb-mcp`. A missing
`KB_MCP__API_KEYS` or `KB_MCP__KB_API__API_KEY` fails at startup with the
variable name — leave those lines out of `.env.prod` rather than blank, then
set them; a blank `KB_MCP__…API_KEY=` is rejected the same way a blank
provider key is.

**`:8443` connection refused, `:443` works.** `tailscale serve status` inside
the proxy should show both handlers. A serve.json that still has only `443`
is a checkout from before Phase 11; pull and recreate the tailscale
container so it remounts `deploy/tailscale-serve.json`.

**MCP host gets 421 Invalid Host header.** That is the SDK's DNS-rebinding
middleware. Production passes it disabled on purpose (AD-027). If you see
this, the running image is older than that decision.

**MCP host connects but every tool is 401 against `kb-api`.** The outbound
key (`KB_MCP__KB_API__API_KEY`) does not match any entry in
`KB_API__API_KEYS`, or it lacks `search` / `write`. They are two
credentials. Check `logs kb-api` for the grant line and `logs kb-mcp` for
`kb_api=http://kb-api:8000`.
