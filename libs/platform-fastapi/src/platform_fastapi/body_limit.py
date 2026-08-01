"""The application-layer request body cap (Technical Design §8).

Nginx caps bodies too, but the app cannot depend on that: it is reachable
directly in development, from the CLI over a local socket, and from whatever
ends up in front of it later. A limit enforced in one place holds only where
that place is.

The cap is enforced twice for one reason. `Content-Length` is a claim, checked
before a single body byte is read — the cheap path, and the one every JSON
client takes. A chunked request makes no such claim, so the bytes are counted as
they arrive and the request is refused the moment the running total passes the
ceiling, rather than after the whole thing has been buffered into memory.
Trusting the header alone would leave the ceiling unenforced for exactly the
caller who declines to state a size.

**Refusal cannot be an exception.** The obvious implementation raises from the
wrapped `receive` and catches it here, and it does not work: FastAPI wraps
*anything* thrown while it parses a body into `HTTPException(400, "There was an
error parsing the body")`, so a `413` arrives at the client as a `400` about
malformed JSON. Instead the wrapped `receive` reports `http.disconnect` — which
every ASGI framework already knows how to unwind from — whatever the app
answers with is dropped, and this middleware sends the `413` itself. Dropping
the app's response is safe precisely because it is the response to a request
that was never fully delivered.
"""

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from platform_core import PayloadTooLargeError, get_logger
from platform_fastapi.problem import problem_response

__all__ = ["BodySizeLimitMiddleware"]

_logger = get_logger("platform.http")

# Methods that cannot carry a body worth capping. Checking the header on a GET
# would reject a client that spells `Content-Length: 0` in an unusual way.
_BODYLESS = frozenset({"GET", "HEAD", "DELETE", "OPTIONS"})

_DISCONNECT: Message = {"type": "http.disconnect"}


class BodySizeLimitMiddleware:
    """Refuse request bodies over `max_bytes` with `413`."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in _BODYLESS:
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._refuse(scope, receive, send, received=declared)
            return

        counted = 0
        exceeded = False
        response_started = False

        async def counting_receive() -> Message:
            nonlocal counted, exceeded
            if exceeded:
                return _DISCONNECT
            message = await receive()
            if message["type"] == "http.request":
                counted += len(message.get("body", b""))
                if counted > self.max_bytes:
                    exceeded = True
                    return _DISCONNECT
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if exceeded and not response_started:
                # Whatever the app made of a truncated body, the client is
                # getting the `413` instead.
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

        if exceeded and not response_started:
            await self._refuse(scope, receive, send, received=counted)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send, *, received: int) -> None:
        _logger.warning(
            "request_body_too_large",
            path=scope["path"],
            limit_bytes=self.max_bytes,
            received_bytes=received,
        )
        error = PayloadTooLargeError(
            f"Request body exceeds the {self.max_bytes} byte limit.",
            context={"limit_bytes": self.max_bytes},
        )
        response = problem_response(
            Request(scope, receive),
            status=error.status_code,
            title=error.title,
            code=error.code,
            detail=error.detail,
            extensions=error.context,
        )
        await response(scope, receive, send)


def _declared_length(scope: Scope) -> int | None:
    raw = Headers(scope=scope).get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        # A malformed header is the protocol layer's problem, not the cap's;
        # inventing a 413 for it would answer the wrong question.
        return None
