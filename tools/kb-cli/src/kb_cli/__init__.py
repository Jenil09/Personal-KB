"""`kb-cli` — the operator client for `kb-api` (AD-025).

An HTTP client of a deployed service, installed with `uv tool install` and run
from a laptop over Tailscale (AD-023). It imports `platform-core` for the error
hierarchy and the settings base and nothing else from the workspace: it must be
able to be older than the service it talks to.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
