"""RFC 9457 problem+json — the only place a failure becomes an HTTP response.

Services raise `PlatformError`; nothing constructs `HTTPException` (Technical
Design §5). Four handlers cover everything that can reach the client:

* `PlatformError`      — the expected failures, mapped by their own status and code
* `RequestValidationError` — FastAPI's own 422, restated as problem+json
* `HTTPException`      — what routing itself raises: 404, 405, and 415
* anything else        — caught in `RequestContextMiddleware`, logged, and answered
  as a bare 500 that leaks nothing

`type` stays `about:blank` because there is no documentation URI to dereference;
RFC 9457 then requires `title` to be the status phrase, which is exactly what the
error classes carry. Clients branch on the `code` extension member instead — it
is stable API contract and is stored on every audit row.
"""

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response

from platform_core import PlatformError, get_logger, get_request_id

__all__ = ["PROBLEM_CONTENT_TYPE", "install_error_handlers", "problem_response"]

PROBLEM_CONTENT_TYPE = "application/problem+json"

REQUEST_ID_STATE_ATTR = "request_id"

# Members RFC 9457 defines. An error's `context` is merged in alongside them as
# extension members, so it must not overwrite these.
_RESERVED = frozenset({"type", "title", "status", "detail", "instance", "code", "request_id"})

_logger = get_logger("platform.http")


def problem_response(
    request: Request,
    *,
    status: int,
    title: str,
    code: str,
    detail: str,
    extensions: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build a problem+json response for `request`."""
    # `Any` because extension members are arbitrary JSON-serialisable values.
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
        "request_id": request_id_of(request),
    }
    body.update(
        {
            key: jsonable_encoder(value)
            for key, value in (extensions or {}).items()
            if key not in _RESERVED
        }
    )

    response_headers = dict(headers or {})
    if status == HTTPStatus.UNAUTHORIZED:
        response_headers.setdefault("WWW-Authenticate", "Bearer")
    return JSONResponse(
        body,
        status_code=status,
        media_type=PROBLEM_CONTENT_TYPE,
        headers=response_headers or None,
    )


def request_id_of(request: Request) -> str:
    """The request ID bound by `RequestContextMiddleware`.

    The scope is read first because it outlives the bound context: a response
    built after that context unwinds — starlette's own `ServerErrorMiddleware`,
    for one — still reports the ID the caller was given.
    """
    scoped: str | None = getattr(request.state, REQUEST_ID_STATE_ATTR, None)
    return scoped or get_request_id() or ""


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(PlatformError, _handle_platform_error)
    app.add_exception_handler(RequestValidationError, _handle_request_validation_error)
    app.add_exception_handler(HTTPException, _handle_http_exception)


def _handle_platform_error(request: Request, exc: Exception) -> Response:
    # Narrowing is safe: starlette only routes here what was registered above.
    error = cast(PlatformError, exc)
    event = "request_failed"
    if error.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        _logger.error(event, code=error.code, status=error.status_code, exc_info=error)
    else:
        _logger.warning(event, code=error.code, status=error.status_code, detail=error.detail)
    return problem_response(
        request,
        status=error.status_code,
        title=error.title,
        code=error.code,
        detail=error.detail,
        extensions=error.context,
    )


def _handle_request_validation_error(request: Request, exc: Exception) -> Response:
    error = cast(RequestValidationError, exc)
    _logger.warning("request_invalid", errors=len(error.errors()))
    return problem_response(
        request,
        status=HTTPStatus.UNPROCESSABLE_ENTITY,
        title=HTTPStatus.UNPROCESSABLE_ENTITY.phrase,
        code="validation_error",
        detail="The request body did not match the expected schema.",
        extensions={"errors": error.errors()},
    )


def _handle_http_exception(request: Request, exc: Exception) -> Response:
    """Restate starlette's own failures — 404, 405, 415 — as problem+json."""
    error = cast(HTTPException, exc)
    status = HTTPStatus(error.status_code)
    return problem_response(
        request,
        status=error.status_code,
        title=status.phrase,
        code=status.name.lower(),
        detail=str(error.detail),
        headers=error.headers,
    )
