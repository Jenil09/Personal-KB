"""`python -m kb_mcp` — the uvicorn entry point.

One worker, like `kb-api` (AD-015): the session manager holds transport state in
process memory, so a second worker would answer requests for sessions it has
never heard of.

Configuration failures are turned into the sentence that names the environment
variable to set, rather than a pydantic traceback — an operator reading a
container's first ten lines of output should not have to parse one.
"""

import sys

import uvicorn

from kb_mcp.config import get_settings
from kb_mcp.main import build_app
from platform_core import ConfigurationError, configure_logging


def main() -> int:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        return 2

    configure_logging(level=settings.log_level, json_output=settings.log_json)
    uvicorn.run(
        build_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
