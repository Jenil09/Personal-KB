"""`/health` — the shape the PRD §6.1 promises, and the budget it promises it in."""

import asyncio
import time

import httpx
import pytest

from platform_core import ConfigurationError
from platform_fastapi import CheckResult, HealthCheck, create_health_router


def _check(name: str, result: CheckResult) -> HealthCheck:
    async def probe() -> CheckResult:
        return result

    return HealthCheck(name=name, probe=probe)


def _slow_check(name: str, seconds: float) -> HealthCheck:
    async def probe() -> CheckResult:
        await asyncio.sleep(seconds)
        return CheckResult.ok()

    return HealthCheck(name=name, probe=probe)


def _broken_check(name: str) -> HealthCheck:
    async def probe() -> CheckResult:
        raise ConnectionRefusedError("postgres is not listening")

    return HealthCheck(name=name, probe=probe)


async def test_a_service_with_no_dependencies_is_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.2"}


async def test_healthy_dependencies_report_their_own_word(make_app, client_for) -> None:
    checks = [
        _check("chromadb", CheckResult.ok()),
        _check("postgres", CheckResult.ok()),
        _check("audit_spill", CheckResult.ok("empty")),
    ]
    async with client_for(make_app(health_checks=checks)) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "1.2",
        "chromadb": "connected",
        "postgres": "connected",
        "audit_spill": "empty",
    }


async def test_a_degraded_dependency_still_serves(make_app, client_for) -> None:
    """A non-empty spill file needs an operator, not a load-balancer removal."""
    checks = [
        _check("postgres", CheckResult.ok()),
        _check("audit_spill", CheckResult.degraded("1204 records pending")),
    ]
    async with client_for(make_app(health_checks=checks)) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["audit_spill"] == "1204 records pending"


async def test_an_unavailable_dependency_answers_503(make_app, client_for) -> None:
    checks = [_check("chromadb", CheckResult.ok()), _broken_check("postgres")]
    async with client_for(make_app(health_checks=checks)) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "version": "1.2",
        "chromadb": "connected",
        "postgres": "unavailable",
    }


async def test_unavailable_outranks_degraded(make_app, client_for) -> None:
    checks = [
        _check("audit_spill", CheckResult.degraded("spilled")),
        _broken_check("postgres"),
    ]
    async with client_for(make_app(health_checks=checks)) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


async def test_a_hanging_check_fails_inside_the_budget(make_app, client_for) -> None:
    checks = [_slow_check("chromadb", seconds=30)]
    app = make_app(health_checks=checks, health_timeout_seconds=0.05)

    started = time.perf_counter()
    async with client_for(app) as client:
        response = await client.get("/health")
    elapsed = time.perf_counter() - started

    assert response.status_code == 503
    assert response.json()["chromadb"] == "timeout"
    assert elapsed < 1.0


async def test_checks_run_concurrently_so_the_budget_is_wall_clock(make_app, client_for) -> None:
    """Three 100 ms checks take ~100 ms, not ~300 ms — adding a dependency
    must not lengthen the probe."""
    checks = [_slow_check(f"dep{index}", seconds=0.1) for index in range(3)]

    started = time.perf_counter()
    async with client_for(make_app(health_checks=checks)) as client:
        response = await client.get("/health")
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 0.25


async def test_one_slow_check_does_not_stop_the_others_reporting(make_app, client_for) -> None:
    checks = [_slow_check("chromadb", seconds=30), _check("postgres", CheckResult.ok())]
    app = make_app(health_checks=checks, health_timeout_seconds=0.05)

    async with client_for(app) as client:
        body = (await client.get("/health")).json()

    assert body == {
        "status": "unavailable",
        "version": "1.2",
        "chromadb": "timeout",
        "postgres": "connected",
    }


@pytest.mark.parametrize("name", ["status", "version"])
def test_a_check_may_not_shadow_a_response_member(name: str) -> None:
    with pytest.raises(ConfigurationError):
        create_health_router(version="1.0", checks=[_check(name, CheckResult.ok())])


def test_check_names_must_be_unique() -> None:
    with pytest.raises(ConfigurationError):
        create_health_router(
            version="1.0",
            checks=[_check("postgres", CheckResult.ok()), _check("postgres", CheckResult.ok())],
        )
