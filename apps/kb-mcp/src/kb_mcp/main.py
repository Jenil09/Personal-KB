"""The composition root — one client, one server, one Starlette app around them.

Built synchronously before the app exists, so the tools are registered with the
client already in hand; the lifespan handler only opens and closes things. That
is `kb_api.main`'s rule, and it is what keeps a tool body from reaching into
`app.state` for its dependencies.

Two ordering constraints, both of which fail confusingly if ignored:

1. **`streamable_http_app()` must be called before `session_manager`.** The SDK
   creates the manager lazily and the property raises until it exists.
2. **The parent lifespan must enter `session_manager.run()`.** A mounted
   sub-application's own lifespan never runs, so without this every request fails
   at dispatch with `RuntimeError: Task group is not initialized` — at request
   time, not at startup, which is the worst place to learn it.

`/health` is process liveness and nothing more. Whether `kb-api` is reachable is
what the `kb_health` *tool* reports; a container healthcheck that failed when an
upstream was down would turn one outage into two restarts.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from kb_client.client import KbClient
from kb_mcp.config import KbMcpSettings, get_settings
from kb_mcp.server import build_server
from platform_core import get_logger

__all__ = ["build_app"]

_logger = get_logger("kb.mcp.startup")


def build_app(
    settings: KbMcpSettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    """Assemble the server.

    `transport` replaces what the `/v1` client sends over, which is the seam the
    suite drives the whole stack through: a request arrives over ASGI, passes
    `RequireAuthMiddleware` and `AuthContextMiddleware`, reaches a tool, and is
    answered from `kb_client.testing.FakeService` rather than from a service. It
    is the same seam `KbClient` already documents, exposed one level up — without
    it the scope checks can only be tested below the transport that sets the
    context they read.
    """
    resolved = settings or get_settings()

    # One client per process, opened here and closed in the lifespan handler.
    # Never per-request: that would pay a TLS handshake on every tool call.
    # Timeouts are already explicit on every `KbClient` call.
    client = KbClient(resolved.kb_api, transport=transport)
    server = build_server(resolved, client)

    # DNS-rebinding protection is host-dependent and is Stage 4's decision, not
    # something to inherit silently: for a loopback host the SDK installs
    # `TransportSecuritySettings` allowing only loopback `Host` headers, and
    # binding `0.0.0.0` turns the protection off rather than adapting it. Left
    # explicit here so the deployment step is a change to this line rather than
    # an omission nobody sees.
    mcp_app = server.streamable_http_app(host=resolved.host)
    session_manager = server.session_manager

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            _logger.info(
                "startup_complete",
                kb_api=resolved.kb_api.base_url,
                grants={
                    key_id: sorted(scopes) if scopes is not None else "unrestricted"
                    for key_id, scopes in _grants(resolved)
                },
                ingest=resolved.allow_ingest,
            )
            try:
                yield
            finally:
                await client.aclose()

    return Starlette(
        routes=[
            Route("/health", _health, methods=["GET"]),
            # Mounted last and at the root: the SDK serves `/mcp` and, when
            # `resource_server_url` is set, the RFC 9728 well-known route. That
            # path carries no `/mcp` suffix — the well-known segment is inserted
            # before the resource path, which is empty here.
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
    )


async def _health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _grants(settings: KbMcpSettings) -> list[tuple[str, frozenset[str] | None]]:
    """Every configured key and what it may do, for the startup log.

    Logged because the permissive default is easy to configure by accident: a key
    absent from `KB_MCP__API_KEY_SCOPES` gets every scope, and that should be
    visible in the first lines of the log rather than at the first call nobody
    expected to succeed.
    """
    return [(key_id, settings.api_key_scopes.get(key_id)) for key_id in sorted(settings.api_keys)]
