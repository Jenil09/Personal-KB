"""Builds the `MCPServer` — auth wired, tools registered, nothing read at import.

A **factory**, not a module-level object, for the reason `kb_api.main` gives at
length: a module that reads the environment at import time is unimportable by a
test that wants to supply its own settings.

The client is passed in rather than built here, so this function has one job and
`main.build_app` keeps ownership of the connection it has to close.
"""

from mcp.server import MCPServer
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl

from kb_client.client import KbClient
from kb_mcp.auth import StaticTokenVerifier
from kb_mcp.config import KbMcpSettings
from kb_mcp.render import UNTRUSTED_NOTICE
from kb_mcp.tools import register_tools
from platform_core import ApiKeyRegistry

__all__ = ["INSTRUCTIONS", "build_server"]

INSTRUCTIONS = f"""\
This server exposes a personal knowledge base: a corpus of notes, architecture \
documents, incident reports, and operating procedures, searchable by meaning.

Use kb_search to answer a question from the corpus — it returns the relevant \
passages. Use kb_get_document only when a whole document is genuinely wanted; \
documents here are large and the body comes back truncated. kb_ingest_document \
adds to the corpus, and the service embeds server-side.

{UNTRUSTED_NOTICE}\
"""


def build_server(settings: KbMcpSettings, client: KbClient) -> MCPServer:
    """The server, with auth enabled and every configured tool attached.

    `required_scopes` is `None` deliberately. The SDK enforces it endpoint-wide
    in `RequireAuthMiddleware`, before a tool is chosen — naming `search` here
    would give a write-only key a `403` on the whole transport and still could
    not express the per-tool split. It is a floor for the endpoint; the scope
    checks that matter are in the tool bodies (`kb_mcp.auth.authorize`).
    """
    server: MCPServer = MCPServer(
        name=settings.service_name,
        version=settings.service_version,
        instructions=INSTRUCTIONS,
        token_verifier=StaticTokenVerifier(
            ApiKeyRegistry(settings.api_keys, settings.api_key_scopes)
        ),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(settings.issuer_url),
            resource_server_url=(
                AnyHttpUrl(settings.resource_server_url) if settings.resource_server_url else None
            ),
            required_scopes=None,
        ),
    )
    register_tools(server, client, settings)
    return server
