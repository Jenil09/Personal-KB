"""`/health` — unversioned, unauthenticated, and bounded (AD-012, Design §5).

Checks run concurrently under one wall-clock budget, so adding a dependency
does not lengthen the probe. A check that hangs is a failed check: the budget
is the whole point, since an unbounded health endpoint takes the orchestrator
down with whatever it is checking.

Three outcomes rather than two, because they call for different reactions:

| overall       | HTTP | meaning                                              |
| ------------- | ---- | ---------------------------------------------------- |
| `ok`          | 200  | everything answered                                   |
| `degraded`    | 200  | still serving, but something needs an operator — a
                         non-empty audit spill file is the case v1 has         |
| `unavailable` | 503  | a dependency is down; take this instance out of rotation |
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from platform_core import ConfigurationError, get_logger

__all__ = ["CheckResult", "HealthCheck", "HealthStatus", "create_health_router"]

_logger = get_logger("platform.health")

# `status` and `version` are the response's own members; a check may not shadow them.
_RESERVED = frozenset({"status", "version"})


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A check's verdict plus the word reported for it, e.g. `connected`."""

    status: HealthStatus
    detail: str

    @classmethod
    def ok(cls, detail: str = "connected") -> "CheckResult":
        return cls(HealthStatus.OK, detail)

    @classmethod
    def degraded(cls, detail: str) -> "CheckResult":
        return cls(HealthStatus.DEGRADED, detail)

    @classmethod
    def unavailable(cls, detail: str = "unavailable") -> "CheckResult":
        return cls(HealthStatus.UNAVAILABLE, detail)


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """A named probe. `name` is the key it reports under in the response body."""

    name: str
    probe: Callable[[], Awaitable[CheckResult]]


def create_health_router(
    *,
    version: str,
    checks: Sequence[HealthCheck] = (),
    timeout_seconds: float = 2.0,
) -> APIRouter:
    names = [check.name for check in checks]
    if collisions := _RESERVED.intersection(names):
        raise ConfigurationError(f"health checks may not be named {sorted(collisions)}")
    if len(set(names)) != len(names):
        raise ConfigurationError(f"health check names must be unique, got {names}")

    router = APIRouter(tags=["health"])

    @router.get(
        "/health",
        summary="Liveness and dependency health",
        responses={
            HTTPStatus.OK: {"description": "Healthy, or degraded but still serving"},
            HTTPStatus.SERVICE_UNAVAILABLE: {"description": "A dependency is unavailable"},
        },
    )
    async def health() -> JSONResponse:
        results = await _run_all(checks, timeout_seconds)
        overall = _overall(results.values())
        # `Any` because the body mixes the fixed members with one string per check.
        body: dict[str, Any] = {"status": overall.value, "version": version}
        body.update({name: result.detail for name, result in results.items()})
        status_code = (
            HTTPStatus.SERVICE_UNAVAILABLE if overall is HealthStatus.UNAVAILABLE else HTTPStatus.OK
        )
        return JSONResponse(body, status_code=status_code)

    return router


async def _run_all(checks: Sequence[HealthCheck], timeout_seconds: float) -> dict[str, CheckResult]:
    results = await asyncio.gather(*(_run(check, timeout_seconds) for check in checks))
    return dict(zip((check.name for check in checks), results, strict=True))


async def _run(check: HealthCheck, timeout_seconds: float) -> CheckResult:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await check.probe()
    except TimeoutError:
        _logger.warning("health_check_timeout", check=check.name, budget=timeout_seconds)
        return CheckResult.unavailable("timeout")
    except Exception as exc:
        _logger.warning("health_check_failed", check=check.name, exc_info=exc)
        return CheckResult.unavailable()


def _overall(results: Iterable[CheckResult]) -> HealthStatus:
    statuses = {result.status for result in results}
    if HealthStatus.UNAVAILABLE in statuses:
        return HealthStatus.UNAVAILABLE
    if HealthStatus.DEGRADED in statuses:
        return HealthStatus.DEGRADED
    return HealthStatus.OK
