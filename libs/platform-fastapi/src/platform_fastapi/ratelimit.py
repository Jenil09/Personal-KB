"""Abuse and runaway-loop detection (AD-014).

Four thresholds, and only two of them reject. That asymmetry is the decision:
the gap between normal traffic (~150 requests/day) and the legitimate schedule
ceiling (~1,080/day) is wide enough that any limit set near normal would
eventually refuse real work, so the middle threshold tells you the day was
unusual instead of deciding it was wrong.

| threshold        | default    | effect                          |
| ---------------- | ---------- | ------------------------------- |
| burst            | 60/minute  | `429`                           |
| daily anomaly    | 600/day    | flags the audit row, serves it  |
| daily ceiling    | 2,000/day  | `429`                           |
| repeat burst     | 5 in 5 min | flags the audit row, serves it  |

**Counters are per key, not per address.** Under AD-023 the two callers arrive
as a Docker-network address and a Tailscale proxy address respectively, so the
proxy's address is shared by everything the operator does and an address-keyed
limit would throttle the CLI on behalf of itself. The key is the identity that
means something here. Unauthenticated traffic has no key, so it is counted
against its address instead — it cannot be allowed to bypass the limiter by
simply not presenting a credential.

**State is in-process.** One Uvicorn worker (AD-015) means one instance, so a
process-local window is the whole story rather than a shard of it. Shared
multi-worker state is on the post-v1 list; adding Redis for a single-worker
deployment would be infrastructure bought against a limit that does not exist.

**Repeat-burst identity includes the payload, and that is why it is detected on
the way out.** The path alone would flag n8n's own workload — ten to fifteen
searches per run all `POST /v1/search` — as a runaway loop within a minute. What
distinguishes a loop from a busy run is the *same* query repeating, which is
known only once the route has recorded what it was asked for. The flag rejects
nothing, so computing it late costs nothing.
"""

import hashlib
import json
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from platform_core import RateLimitedError, get_logger
from platform_fastapi.auth import ApiKeyRegistry
from platform_fastapi.problem import problem_response

__all__ = [
    "ANOMALY_STATE_ATTR",
    "BurstDetector",
    "RateLimitDecision",
    "RateLimitMiddleware",
    "RateLimitSettings",
    "RateLimiter",
    "burst_fingerprint",
]

ANOMALY_STATE_ATTR = "audit_anomaly"
"""Where the daily-anomaly flag lands, for the tier-1 audit row (AD-014)."""

_logger = get_logger("platform.ratelimit")

_MINUTE = 60.0
_DAY = 86_400.0


class RateLimitSettings(BaseModel):
    """AD-014's thresholds, every one of them configurable.

    The defaults are the numbers the decision argued for against a measured
    baseline. They are settings rather than constants because the baseline is an
    observation about one deployment's traffic, and a second service inheriting
    this library will have its own.
    """

    model_config = {"frozen": True}

    enabled: bool = True

    # ~4x the real peak burst of 10-15 per n8n run. Catches a hot loop within
    # seconds without touching a legitimate run.
    per_minute: int = Field(default=60, ge=1)

    # Above normal, below the legitimate schedule ceiling. Flags, never rejects.
    daily_anomaly: int = Field(default=600, ge=1)

    # Comfortably above the ~1,080 legitimate maximum. Pure runaway protection.
    daily_limit: int = Field(default=2000, ge=1)

    repeat_burst_count: int = Field(default=5, ge=2)
    repeat_burst_window_seconds: float = Field(default=300.0, gt=0)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """The outcome of admitting one request."""

    anomaly: bool = False
    """Past the daily anomaly threshold. Recorded on the row; not a rejection."""


class RateLimiter:
    """Sliding windows per identity, evaluated before the request is served."""

    def __init__(self, settings: RateLimitSettings | None = None) -> None:
        self._settings = settings or RateLimitSettings()
        self._events: dict[str, deque[float]] = {}

    def check(self, identity: str) -> RateLimitDecision:
        """Admit or reject one request. Raises `RateLimitedError` to reject.

        The event is recorded only when the request is admitted. A rejected
        request that still counted would extend its own penalty every time the
        caller retried, so a loop that tripped the limit could never fall back
        under it.
        """
        if not self._settings.enabled:
            return RateLimitDecision()

        now = time.monotonic()
        window = self._events.setdefault(identity, deque())
        _evict(window, now - _DAY)

        minute_count = sum(1 for stamp in window if stamp > now - _MINUTE)
        if minute_count >= self._settings.per_minute:
            raise self._reject(identity, "per_minute", minute_count, _MINUTE)
        if len(window) >= self._settings.daily_limit:
            raise self._reject(identity, "daily_limit", len(window), _DAY)

        window.append(now)
        if len(window) == self._settings.daily_anomaly:
            # Logged once, on the crossing, rather than on every request past
            # it — an anomaly that reprints 1,400 times is not a signal.
            _logger.warning(
                "rate_limit_anomaly",
                identity=identity,
                count=len(window),
                threshold=self._settings.daily_anomaly,
            )
        return RateLimitDecision(anomaly=len(window) >= self._settings.daily_anomaly)

    def _reject(self, identity: str, limit: str, count: int, window: float) -> RateLimitedError:
        _logger.warning("rate_limited", identity=identity, limit=limit, count=count)
        retry_after = round(window)
        return RateLimitedError(
            "Too many requests. Slow down and retry after the interval given.",
            context={"limit": limit, "retry_after_seconds": retry_after},
            headers={"Retry-After": str(retry_after)},
        )


class BurstDetector:
    """Flags the same request repeating — a runaway loop, not a busy run."""

    def __init__(self, settings: RateLimitSettings | None = None) -> None:
        self._settings = settings or RateLimitSettings()
        self._seen: dict[str, deque[float]] = {}

    def observe(self, fingerprint: str) -> bool:
        """Record one occurrence. True once the window holds enough of them.

        Stays true for every further repeat inside the window rather than firing
        once on the crossing: the flag answers "was this request part of a
        loop?", and the tenth identical call is no less part of it than the
        fifth.
        """
        if not self._settings.enabled:
            return False
        now = time.monotonic()
        window = self._seen.setdefault(fingerprint, deque())
        _evict(window, now - self._settings.repeat_burst_window_seconds)
        window.append(now)
        return len(window) >= self._settings.repeat_burst_count


def burst_fingerprint(
    identity: str, method: str, path: str, payload: Mapping[str, Any] | None
) -> str:
    """What makes two requests "the same one" for repeat detection.

    The payload is included because the path is not discriminating: every search
    is `POST /v1/search`, so a path-only fingerprint would report n8n's normal
    workload as a loop. `sort_keys` because two callers may serialise the same
    filters in different orders, and `default=str` because the payload is
    whatever the route recorded — a UUID or a datetime in it should change the
    fingerprint, not raise.
    """
    body = json.dumps(payload or {}, sort_keys=True, default=str)
    return hashlib.sha256(f"{identity}\0{method}\0{path}\0{body}".encode()).hexdigest()


def _evict(window: deque[float], cutoff: float) -> None:
    while window and window[0] <= cutoff:
        window.popleft()


class RateLimitMiddleware:
    """Applies the limiter before the request reaches the router.

    Placed inside `AuditMiddleware` so that a `429` is itself a recorded
    request — the burst that tripped the limit is the traffic the trail most
    needs, and a limiter that rejected requests before they were logged would
    erase the incident it exists to detect (AD-013, AD-014).

    Resolving the key here rather than reusing the router's dependency is
    deliberate: the dependency runs after routing, and a request to an unknown
    path must still be counted. The comparison is a handful of `compare_digest`
    calls, which is nothing beside the request it guards.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        registry: ApiKeyRegistry,
        exempt_paths: Sequence[str] = (),
    ) -> None:
        self.app = app
        self._limiter = limiter
        self._registry = registry
        self._exempt = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._exempt:
            await self.app(scope, receive, send)
            return

        state: dict[str, Any] = scope.setdefault("state", {})
        identity = self._identity(scope)
        try:
            decision = self._limiter.check(identity)
        except RateLimitedError as exc:
            state[ANOMALY_STATE_ATTR] = False
            response = problem_response(
                Request(scope, receive),
                status=exc.status_code,
                title=exc.title,
                code=exc.code,
                detail=exc.detail,
                extensions=exc.context,
                headers=exc.headers,
            )
            await response(scope, receive, send)
            return

        state[ANOMALY_STATE_ATTR] = decision.anomaly
        await self.app(scope, receive, send)

    def _identity(self, scope: Scope) -> str:
        """Whose counter this request belongs to.

        A recognised key wins. Failing that the peer address, so a caller cannot
        opt out of the limiter by presenting no credential at all — that traffic
        is exactly what a limiter is for.
        """
        scheme, separator, credential = Headers(scope=scope).get("authorization", "").partition(" ")
        if separator and scheme.lower() == "bearer" and credential.strip():
            principal = self._registry.identify(credential.strip())
            if principal is not None:
                return f"key:{principal.key_id}"
        client = scope.get("client")
        return f"ip:{client[0]}" if client else "anonymous"
