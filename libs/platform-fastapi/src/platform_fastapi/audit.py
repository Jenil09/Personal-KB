"""Tier-1 audit wiring — one row per request, whatever happens (AD-013).

Phase 3 built the writer, the spill file, and the reconciler. This is the thing
that calls them, and the exit criterion it has to satisfy is a counting one:
requests in, rows out, equal. Not "the happy path is logged".

That is why this is ASGI middleware rather than a dependency. A dependency runs
only once routing and authentication have succeeded, so it would miss precisely
the requests worth reconstructing later — the rejected key, the unknown path,
the burst that tripped the limiter, the body over the cap. Middleware sees all
of them because it sits above the router.

**Placement.** Inside `RequestContextMiddleware`, because the record needs the
correlation ID that middleware binds. Outside the body cap, the rate limiter,
and the router, because each of those produces a response this has to record.
`create_app` fixes the order; it is not a per-service choice.

**An unhandled exception is recorded and re-raised.** The 500 itself is built
one layer out, so this never sees its status — it stamps `500` and
`server_error` from the exception and lets it continue. Swallowing it here would
produce an audit row describing a response the caller never got.

**Operation and payload come from the route, not from here.** AD-013 wants the
search query text and the ingest's title, source, hash, and size on the row, and
that is service knowledge — reading it here would mean buffering and re-parsing
a body the router has already parsed. Routes call `record_operation()` instead,
which is a state write the middleware picks up on the way out.

`platform-db` is a dependency of this library for one reason: `AuditRecord` is
the shared vocabulary AD-018 deliberately centralised, so that the writer, the
spill format, and the columns cannot drift. Restating it at this boundary to
avoid the import would reintroduce exactly the drift AD-018 exists to prevent.
"""

import time
from collections.abc import Callable, Mapping
from http import HTTPStatus
from ipaddress import ip_address
from typing import Any
from uuid import UUID

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from platform_core import get_logger
from platform_db import AuditRecord, AuditTrail, Outcome, fingerprint_credential
from platform_fastapi.auth import PRINCIPAL_STATE_ATTR, Principal
from platform_fastapi.problem import ERROR_CODE_STATE_ATTR, REQUEST_ID_STATE_ATTR
from platform_fastapi.ratelimit import ANOMALY_STATE_ATTR, BurstDetector, burst_fingerprint

__all__ = [
    "OPERATION_STATE_ATTR",
    "PAYLOAD_STATE_ATTR",
    "AuditMiddleware",
    "record_operation",
]

OPERATION_STATE_ATTR = "audit_operation"
PAYLOAD_STATE_ATTR = "audit_payload"

_logger = get_logger("platform.audit")


def record_operation(
    request: Request, operation: str, payload: Mapping[str, Any] | None = None
) -> None:
    """Attach the operation detail AD-013 wants on this request's audit row.

    Called from a route handler once it knows what the request actually asked
    for. `payload` is the operation's own shape — a search's query and filters,
    an ingest's title, source, `content_hash`, and byte size — and lands in a
    JSONB column unmodified, so it must be JSON-serialisable.

    Safe to call before the work succeeds, and better to: a request that fails
    halfway is the one whose payload matters most, and a call placed after the
    service returns records nothing when the service raises.
    """
    request.state.__setattr__(OPERATION_STATE_ATTR, operation)
    if payload is not None:
        request.state.__setattr__(PAYLOAD_STATE_ATTR, dict(payload))


class AuditMiddleware:
    """Writes exactly one tier-1 record per HTTP request.

    `observer` is where tier-2 hangs off tier-1 without duplicating dispatch.
    The record is already built here and the exception, if there was one, is
    already in hand — so a service wanting an `error_logs` row per failure needs
    a callback rather than a second pass over the same information. It is
    synchronous, and its failures are swallowed for the same reason tier 2 is
    droppable: it must not be able to affect the response or the tier-1 write.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        trail: AuditTrail,
        bursts: BurstDetector | None = None,
        observer: Callable[[AuditRecord, BaseException | None], None] | None = None,
    ) -> None:
        self.app = app
        self._trail = trail
        self._bursts = bursts
        self._observer = observer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state: dict[str, Any] = scope.setdefault("state", {})
        started = time.perf_counter()
        status = HTTPStatus.INTERNAL_SERVER_ERROR.value
        failure: BaseException | None = None

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except BaseException as exc:  # recorded, then re-raised unchanged
            failure = exc
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000)
            await self._write(scope, state, status=status, latency_ms=latency_ms, failure=failure)

        if failure is not None:
            raise failure

    async def _write(
        self,
        scope: Scope,
        state: Mapping[str, Any],
        *,
        status: int,
        latency_ms: int,
        failure: BaseException | None,
    ) -> None:
        request_id = state.get(REQUEST_ID_STATE_ATTR)
        if not isinstance(request_id, str):
            # `RequestContextMiddleware` is unconditionally outside this one, so
            # this means the stack was assembled by hand and wrongly. Log rather
            # than raise: losing the request over an audit defect would be the
            # fail-closed behaviour AD-013 explicitly rejects.
            _logger.error("audit_skipped_without_request_id", path=scope.get("path"))
            return

        headers = Headers(scope=scope)
        candidate = state.get(PRINCIPAL_STATE_ATTR)
        principal = candidate if isinstance(candidate, Principal) else None
        authenticated = principal is not None
        payload = _payload(state, headers, authenticated=authenticated)

        record = AuditRecord(
            request_id=UUID(request_id),
            method=scope["method"],
            path=scope["path"],
            status_code=status,
            outcome=_outcome(status, failure),
            latency_ms=latency_ms,
            key_id=principal.key_id if principal is not None else None,
            client_ip=_client_ip(scope),
            user_agent=headers.get("user-agent"),
            error_code=_error_code(state, failure),
            operation=_operation(state, authenticated=authenticated),
            payload=payload,
            repeat_burst=self._repeat_burst(scope, principal, payload),
            anomaly=bool(state.get(ANOMALY_STATE_ATTR, False)),
        )
        await self._trail.record(record)
        self._observe(record, failure)

    def _observe(self, record: AuditRecord, failure: BaseException | None) -> None:
        if self._observer is None:
            return
        try:
            self._observer(record, failure)
        except Exception as exc:
            # Tier 2 is best-effort by contract. A defect in the observer must
            # not become a defect in the trail that is guaranteed.
            _logger.warning("audit_observer_failed", exc_info=exc)

    def _repeat_burst(
        self, scope: Scope, principal: Principal | None, payload: Mapping[str, Any] | None
    ) -> bool:
        """Whether this request is one of several identical ones (AD-014).

        Evaluated here rather than in the limiter because the fingerprint needs
        the payload the route recorded, and the route runs after the limiter.
        The flag rejects nothing, so nothing is lost by knowing it late — see
        `ratelimit.py` for why the path alone will not do.
        """
        if self._bursts is None:
            return False
        identity = principal.key_id if principal is not None else _client_ip(scope) or ""
        return self._bursts.observe(
            burst_fingerprint(identity, scope["method"], scope["path"], payload)
        )


def _outcome(status: int, failure: BaseException | None) -> Outcome:
    """The forensic filter (Design §2.2), derived from what the caller saw.

    `401` and `429` get their own outcomes rather than being folded into
    `client_error`, because "who was refused" and "who was throttled" are the two
    questions the trail is most often asked and neither should need a status-code
    filter to answer.
    """
    if failure is not None:
        return Outcome.SERVER_ERROR
    if status == HTTPStatus.UNAUTHORIZED:
        return Outcome.AUTH_FAILED
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return Outcome.RATE_LIMITED
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return Outcome.SERVER_ERROR
    if status >= HTTPStatus.BAD_REQUEST:
        return Outcome.CLIENT_ERROR
    return Outcome.SUCCESS


def _error_code(state: Mapping[str, Any], failure: BaseException | None) -> str | None:
    """The problem+json `code` the caller received.

    Stamped by `problem_response`, which is the single place an error becomes a
    response — so this stays correct without the middleware parsing a body it
    would have to buffer to read.
    """
    if failure is not None:
        return "internal_error"
    code = state.get(ERROR_CODE_STATE_ATTR)
    return code if isinstance(code, str) else None


def _operation(state: Mapping[str, Any], *, authenticated: bool) -> str | None:
    operation = state.get(OPERATION_STATE_ATTR)
    if isinstance(operation, str):
        return operation
    # A request that never reached a handler still gets a name, so the rejected
    # traffic in the trail is filterable rather than a block of NULLs.
    return None if authenticated else "auth_failure"


def _payload(
    state: Mapping[str, Any], headers: Headers, *, authenticated: bool
) -> dict[str, Any] | None:
    payload = state.get(PAYLOAD_STATE_ATTR)
    if isinstance(payload, dict):
        return payload
    if authenticated:
        return None
    # AD-013: a failed authentication records the presented credential
    # fingerprinted, never stored. Enough to see one bad key retried a thousand
    # times; not enough to recover it.
    return _presented_fingerprint(headers)


def _presented_fingerprint(headers: Headers) -> dict[str, Any] | None:
    scheme, separator, credential = headers.get("authorization", "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not credential.strip():
        return None
    return {"key_fingerprint": fingerprint_credential(credential.strip())}


def _client_ip(scope: Scope) -> str | None:
    """The peer address, or `None` when it is not one.

    `X-Forwarded-For` is deliberately not read. Under AD-023 the two callers
    reach this service over a Docker network and a Tailscale proxy respectively,
    and trusting a forwarded header without knowing what set it is how an
    attacker writes any address they like into the trail. The column is `INET`;
    an ASGI transport's `"testclient"` host is not one, and a made-up value would
    be worse than none.
    """
    client = scope.get("client")
    if not client:
        return None
    try:
        return str(ip_address(client[0]))
    except ValueError:
        return None
