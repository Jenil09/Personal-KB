"""`kb-client` — the `/v1` contract of `kb-api`, and the client that speaks it.

Extracted from `kb-cli` when `kb-mcp` became a second consumer. It imports
`platform-core` for the error hierarchy and the settings base and nothing else
from the workspace, because everything built on it is an HTTP client of a
deployed service (AD-025) and must be able to be older than that service.

What belongs here is the wire: the response models, the client, the controlled
vocabulary a suggestion is answered in, and the fake that serves the contract in
tests. What does not is anything an operator sees — rendering, the config file,
the terminal — which stays with the tool that renders it.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
