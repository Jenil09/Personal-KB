"""Request context: correlation ID in and out, access log, catch-all.

Pure ASGI rather than `BaseHTTPMiddleware` because this wraps every request and
`BaseHTTPMiddleware` pays for an anyio task group per call to give back a
convenience this does not need.

The catch-all for unhandled exceptions lives here rather than in an
`add_exception_handler(Exception, ...)`, because starlette's
`ServerErrorMiddleware` sits *outside* every user middleware and re-raises after
responding: a response built there would carry neither the `X-Request-ID` header
nor the bound log context, both having already unwound. Handling it here means
the 500 looks like every other error.
"""

import time
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from platform_core import REQUEST_ID_HEADER, get_logger, request_context
from platform_fastapi.auth import PRINCIPAL_STATE_ATTR, Principal
from platform_fastapi.problem import REQUEST_ID_STATE_ATTR, problem_response

__all__ = ["RequestContextMiddleware"]

_logger = get_logger("platform.http")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # A caller-supplied ID that is not a UUID is replaced rather than
        # propagated — the audit column is a Postgres `UUID` (Design §2.2).
        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        state: dict[str, Any] = scope.setdefault("state", {})
        started = time.perf_counter()

        with request_context(incoming) as request_id:
            state[REQUEST_ID_STATE_ATTR] = request_id
            status = 0
            response_started = False

            async def send_with_request_id(message: Message) -> None:
                nonlocal status, response_started
                if message["type"] == "http.response.start":
                    status = message["status"]
                    response_started = True
                    MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
                await send(message)

            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception as exc:
                _logger.error("request_unhandled_error", exc_info=exc)
                if response_started:
                    # Headers are already on the wire; the server tears the
                    # connection down rather than us contradicting ourselves.
                    raise
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                response = problem_response(
                    Request(scope, receive),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    title=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                    code="internal_error",
                    detail="The request could not be completed. Quote the request_id.",
                )
                await response(scope, receive, send_with_request_id)
            finally:
                _log_access(scope, state, status=status, started=started)


def _log_access(scope: Scope, state: Mapping[str, Any], *, status: int, started: float) -> None:
    principal = state.get(PRINCIPAL_STATE_ATTR)
    client = scope.get("client")
    log = _logger.info
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        log = _logger.error
    elif status >= HTTPStatus.BAD_REQUEST:
        log = _logger.warning

    log(
        "request",
        method=scope["method"],
        path=scope["path"],
        status=status,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        client_ip=client[0] if client else None,
        key_id=principal.key_id if isinstance(principal, Principal) else None,
    )
