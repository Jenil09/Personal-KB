"""Configuration — `KB_MCP__*`, with the connection to `kb-api` nested under it.

Its own prefix for AD-025's reason: `KB_API__*` configures the service being
called, and a shell holding both must not cross-configure. Secrets carry no
defaults, so a missing one fails when settings are constructed at startup rather
than at the first tool call.

**`KbApiConnection` overrides the prefix, and that is load-bearing.** A
`BaseSettings` subclass used as a field runs its own environment source during
validation, so with the inherited `KB_CLIENT__` prefix a bare `KB_CLIENT__API_KEY`
in any shell would populate `kb_api.api_key` — not overriding
`KB_MCP__KB_API__API_KEY`, which wins, but silently *filling in* for it when the
operator set only `KB_MCP__KB_API__BASE_URL`. Measured, not assumed: with the
prefix overridden to `KB_MCP__KB_API__`, both routes read the same variable and
the leak is closed. The inbound key format is `platform_core`'s `ApiKeys` /
`ApiKeyScopes` annotation, so `KB_MCP__API_KEYS` and `KB_API__API_KEYS` cannot
drift apart.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from kb_client.settings import KbClientSettings
from kb_mcp import __version__
from platform_core import ApiKeys, ApiKeyScopes, BaseServiceSettings

__all__ = ["KbApiConnection", "KbMcpSettings", "get_settings"]


class KbApiConnection(KbClientSettings):
    """Where `kb-api` is and which key reaches it — `KB_MCP__KB_API__*`.

    Nothing is added to `KbClientSettings`; only the prefix changes, so that the
    nested model's own source and the parent's nested delimiter name the same
    environment variable instead of two.
    """

    model_config = SettingsConfigDict(env_prefix="KB_MCP__KB_API__")

    user_agent: str = f"kb-mcp/{__version__}"
    """Overrides the base's `None` so MCP traffic is attributable.

    `kb-api` records the header on every tier-1 audit row, and `key_id` alone
    cannot separate this server's calls from any other holder of the same key.
    """


class KbMcpSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_MCP__", env_nested_delimiter="__")

    service_name: str = "kb-mcp"
    service_version: str = "0.1.0"

    kb_api: KbApiConnection

    # `KB_MCP__API_KEYS=claude:s3cr3t` — the tokens an MCP *host* presents to
    # this server, which are not the key this server presents to `kb-api`. Same
    # format as `KB_API__API_KEYS` because it is the same annotated type.
    api_keys: ApiKeys

    # `KB_MCP__API_KEY_SCOPES=claude:search|write` (AD-024). A key named here is
    # held to those scopes; a key *not* named here gets every scope, which is
    # `kb-api`'s permissive default and is chosen for its reason — an unlisted
    # key having no scopes makes a first deploy fail on a valid credential.
    api_key_scopes: ApiKeyScopes = Field(default_factory=dict)

    # Every interface, because a container that binds loopback is unreachable
    # from the compose network. AD-023 is what keeps it off the host: the
    # production stack publishes no ports at all.
    host: str = "0.0.0.0"
    port: int = Field(default=9000, gt=0, lt=65536)

    max_document_chars: int = Field(default=40_000, gt=0)
    """The ceiling `kb_get_document` truncates to.

    `DocumentSummary` puts the corpus at ~0.5 MB a file — roughly 125k tokens,
    which is most of a context window for one tool result. 40k characters is
    about 10k tokens: enough to read a document, small enough that reading three
    is not the end of the conversation.
    """

    allow_ingest: bool = True
    """Whether `kb_ingest_document` is registered at all.

    Off means the tool does not appear in `tools/list`, which is a stronger and
    more legible statement than a tool that exists and always refuses.
    """

    issuer_url: str = "https://kb-mcp.invalid"
    """Nominal, and required by the SDK's `AuthSettings`.

    Tokens here are static keys, so there is no authorization server to name and
    this resolves to nothing on purpose. A host that follows the RFC 9728
    advertisement rather than using a configured `Authorization` header will fail
    against it — which is the documented, legible failure, and a real OAuth
    server is a later phase's problem.
    """

    resource_server_url: str | None = None
    """This server's externally reachable base URL, published in RFC 9728 metadata.

    Unset suppresses the `/.well-known/oauth-protected-resource` route entirely.
    That is right locally, where there is no external URL to advertise, and wrong
    in production: a deployment sets it to the tailnet address so a host that
    *does* read the metadata fails where it can be diagnosed rather than
    silently. Claude Code never reads it either way — a configured
    `Authorization` header short-circuits discovery.
    """


@lru_cache
def get_settings() -> KbMcpSettings:
    """Read once per process. The environment does not change under a server."""
    return KbMcpSettings()
