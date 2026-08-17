"""`kb-mcp` — the knowledge base as MCP tools, over streamable HTTP.

A third consumer of `kb-api` beside n8n and `kb-cli`, and an HTTP client of it
like `kb-cli` is (AD-025): never Postgres, never Chroma. That boundary buys two
things — every tool call lands in the tier-1 audit trail (AD-013), and no caller
can remove a document while leaving its vectors behind.

Nothing here embeds. AD-006 binds the index to one model and one collection, so a
vector produced anywhere but `kb-api` has nowhere to go; `kb_ingest_document`
causes the *service* to embed, and this package holds no provider SDK and no key
for one.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
