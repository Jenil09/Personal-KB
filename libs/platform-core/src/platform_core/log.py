"""structlog configuration: JSON to stdout, one line per event.

Called once from a service's composition root, before anything logs. Human
output is available for local work (`log_json=False`), but production is JSON
so the lines are greppable and parseable by whatever reads them next.

Module is `log`, not `logging`, so it never shadows the standard library for
anything importing it.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger, Processor

__all__ = ["configure_logging", "get_logger"]


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog. Idempotent — the last call wins."""
    renderer: Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_level_number(level)),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Third-party libraries log through the standard library; keep them at the
    # same threshold so raising the level actually quietens the process.
    logging.getLogger().setLevel(_level_number(level))


def get_logger(name: str | None = None, **initial_values: Any) -> FilteringBoundLogger:
    """A bound logger. `**initial_values` are stamped on every event it emits."""
    # `Any` because bound values are arbitrary structured log fields.
    logger: FilteringBoundLogger = structlog.get_logger(name, **initial_values)
    return logger


def _level_number(level: str) -> int:
    try:
        return logging.getLevelNamesMapping()[level.upper()]
    except KeyError:
        raise ValueError(f"unknown log level: {level!r}") from None
