"""Correlation-ID generation and propagation.

One request ID travels through a request: taken from `X-Request-ID` when the
caller supplies one, generated otherwise, bound to a context var so every log
line and audit row picks it up without being passed an argument, and echoed
back in the response header.

The audit trail stores `request_id` as a Postgres `UUID` (Technical Design
§2.2), so a caller-supplied value that is not a UUID is replaced rather than
propagated — an unparseable header must not be able to fail an insert.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID, uuid4

import structlog

__all__ = [
    "REQUEST_ID_HEADER",
    "coerce_request_id",
    "get_request_id",
    "new_request_id",
    "request_context",
]

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return str(uuid4())


def coerce_request_id(raw: str | None) -> str:
    """Return `raw` canonicalised if it is a UUID, otherwise a fresh ID."""
    if raw is None:
        return new_request_id()
    try:
        return str(UUID(raw.strip()))
    except ValueError:
        return new_request_id()


def get_request_id() -> str | None:
    """The current request ID, or `None` outside a request context."""
    return _request_id.get()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request ID for the duration of the block, then restore.

    Yields the resolved ID so a caller can put it in the response header.
    """
    resolved = coerce_request_id(request_id)
    token = _request_id.set(resolved)
    try:
        with structlog.contextvars.bound_contextvars(request_id=resolved):
            yield resolved
    finally:
        _request_id.reset(token)
