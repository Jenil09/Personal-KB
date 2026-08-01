"""Abuse and runaway-loop detection (AD-014).

The thresholds are asserted at their boundaries rather than somewhere past them,
because an off-by-one here is the difference between rejecting a legitimate
n8n run and admitting a hot loop for another sixty seconds.
"""

import pytest
from fastapi import APIRouter

from platform_core import RateLimitedError
from platform_fastapi import BurstDetector, RateLimiter, RateLimitSettings, burst_fingerprint


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def router() -> APIRouter:
    api = APIRouter()

    @api.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    return api


# --- the limiter itself ---------------------------------------------------


def test_requests_under_the_minute_limit_are_admitted() -> None:
    limiter = RateLimiter(RateLimitSettings(per_minute=3))

    for _ in range(3):
        assert limiter.check("key:n8n").anomaly is False


def test_the_minute_limit_rejects_the_next_one() -> None:
    limiter = RateLimiter(RateLimitSettings(per_minute=3))
    for _ in range(3):
        limiter.check("key:n8n")

    with pytest.raises(RateLimitedError) as caught:
        limiter.check("key:n8n")

    error = caught.value
    assert error.status_code == 429
    assert error.code == "rate_limited"
    assert error.context["limit"] == "per_minute"
    assert error.headers["Retry-After"] == "60"


def test_a_rejected_request_does_not_extend_its_own_penalty() -> None:
    """Counting a rejection would mean a caller retrying in a loop could never
    fall back under the limit, however long it waited."""
    limiter = RateLimiter(RateLimitSettings(per_minute=2))
    limiter.check("key:n8n")
    limiter.check("key:n8n")

    for _ in range(5):
        with pytest.raises(RateLimitedError):
            limiter.check("key:n8n")

    assert len(limiter._events["key:n8n"]) == 2


def test_limits_are_per_identity() -> None:
    limiter = RateLimiter(RateLimitSettings(per_minute=1))
    limiter.check("key:n8n")

    with pytest.raises(RateLimitedError):
        limiter.check("key:n8n")
    assert limiter.check("key:cli").anomaly is False


def test_the_daily_anomaly_flags_without_rejecting() -> None:
    """AD-014's middle threshold: above normal, below the legitimate schedule
    ceiling. It says the day was unusual; it does not decide it was wrong."""
    limiter = RateLimiter(RateLimitSettings(per_minute=1000, daily_anomaly=3, daily_limit=100))

    assert [limiter.check("key:n8n").anomaly for _ in range(5)] == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_the_daily_ceiling_rejects() -> None:
    limiter = RateLimiter(RateLimitSettings(per_minute=1000, daily_anomaly=2, daily_limit=4))
    for _ in range(4):
        limiter.check("key:n8n")

    with pytest.raises(RateLimitedError) as caught:
        limiter.check("key:n8n")
    assert caught.value.context["limit"] == "daily_limit"
    assert caught.value.headers["Retry-After"] == "86400"


def test_disabling_the_limiter_admits_everything() -> None:
    limiter = RateLimiter(RateLimitSettings(enabled=False, per_minute=1))
    for _ in range(50):
        assert limiter.check("key:n8n").anomaly is False


# --- burst detection ------------------------------------------------------


def test_the_burst_flag_raises_on_the_configured_repeat() -> None:
    detector = BurstDetector(RateLimitSettings(repeat_burst_count=3))

    assert [detector.observe("same") for _ in range(4)] == [False, False, True, True]


def test_distinct_fingerprints_do_not_accumulate() -> None:
    detector = BurstDetector(RateLimitSettings(repeat_burst_count=2))

    assert not any(detector.observe(f"query-{index}") for index in range(10))


def test_the_fingerprint_separates_queries_on_one_path() -> None:
    one = burst_fingerprint("n8n", "POST", "/v1/search", {"query": "cilium"})
    two = burst_fingerprint("n8n", "POST", "/v1/search", {"query": "longhorn"})

    assert one != two


def test_the_fingerprint_is_stable_across_key_order() -> None:
    """Two callers may serialise the same filters in different orders; the same
    request must not fingerprint differently because of it."""
    one = burst_fingerprint("n8n", "POST", "/v1/search", {"a": 1, "b": 2})
    two = burst_fingerprint("n8n", "POST", "/v1/search", {"b": 2, "a": 1})

    assert one == two


def test_the_fingerprint_separates_identities() -> None:
    one = burst_fingerprint("n8n", "POST", "/v1/search", {"query": "x"})
    two = burst_fingerprint("cli", "POST", "/v1/search", {"query": "x"})

    assert one != two


def test_the_fingerprint_survives_a_payload_json_cannot_encode() -> None:
    """The payload is whatever a route recorded. A UUID in it should change the
    fingerprint, not raise mid-request."""
    from uuid import uuid4

    identifier = uuid4()
    assert burst_fingerprint("n8n", "DELETE", "/v1/documents", {"id": identifier}) != (
        burst_fingerprint("n8n", "DELETE", "/v1/documents", {"id": uuid4()})
    )


# --- the middleware -------------------------------------------------------


async def test_a_throttled_request_gets_429_with_retry_after(
    make_app, client_for, router, valid_key
) -> None:
    app = make_app(routers=[router], rate_limits=RateLimitSettings(per_minute=2))
    async with client_for(app) as http:
        for _ in range(2):
            assert (await http.get("/v1/ping", headers=_auth(valid_key))).status_code == 200
        response = await http.get("/v1/ping", headers=_auth(valid_key))

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    body = response.json()
    assert body["code"] == "rate_limited"
    assert body["request_id"]


async def test_a_throttled_request_is_still_audited(
    make_app, client_for, router, trail, valid_key
) -> None:
    """The burst that trips the limit is the traffic the trail most needs. A
    limiter that rejected before the audit layer would erase the incident."""
    app = make_app(routers=[router], audit_trail=trail, rate_limits=RateLimitSettings(per_minute=1))
    async with client_for(app) as http:
        await http.get("/v1/ping", headers=_auth(valid_key))
        await http.get("/v1/ping", headers=_auth(valid_key))

    assert len(trail.records) == 2
    throttled = trail.records[-1]
    assert throttled.status_code == 429
    assert throttled.outcome.value == "rate_limited"
    assert throttled.error_code == "rate_limited"


async def test_the_anomaly_flag_reaches_the_audit_row(
    make_app, client_for, router, trail, valid_key
) -> None:
    app = make_app(
        routers=[router],
        audit_trail=trail,
        rate_limits=RateLimitSettings(per_minute=100, daily_anomaly=2, daily_limit=100),
    )
    async with client_for(app) as http:
        for _ in range(3):
            await http.get("/v1/ping", headers=_auth(valid_key))

    assert [record.anomaly for record in trail.records] == [False, True, True]


async def test_health_is_exempt_from_the_limiter(make_app, client_for, router) -> None:
    """Throttling the orchestrator's probe turns a busy minute into a restart."""
    app = make_app(routers=[router], rate_limits=RateLimitSettings(per_minute=1))
    async with client_for(app) as http:
        for _ in range(5):
            assert (await http.get("/health")).status_code == 200


async def test_two_keys_do_not_share_a_budget(make_app, client_for, router, valid_key, other_key):
    app = make_app(routers=[router], rate_limits=RateLimitSettings(per_minute=1))
    async with client_for(app) as http:
        assert (await http.get("/v1/ping", headers=_auth(valid_key))).status_code == 200
        assert (await http.get("/v1/ping", headers=_auth(valid_key))).status_code == 429
        assert (await http.get("/v1/ping", headers=_auth(other_key))).status_code == 200


async def test_unauthenticated_traffic_is_counted_by_address(make_app, client_for, router) -> None:
    """A caller must not be able to opt out of the limiter by presenting no
    credential — that traffic is precisely what a limiter is for."""
    app = make_app(routers=[router], rate_limits=RateLimitSettings(per_minute=2))
    async with client_for(app) as http:
        assert (await http.get("/v1/ping")).status_code == 401
        assert (await http.get("/v1/ping")).status_code == 401
        assert (await http.get("/v1/ping")).status_code == 429


async def test_an_unrecognised_key_still_answers_401_not_429(make_app, client_for, router) -> None:
    """The limiter decides whose counter a request belongs to, not whether it
    authenticates. A bad key inside its budget gets the 401 it has coming."""
    app = make_app(routers=[router], rate_limits=RateLimitSettings(per_minute=10))
    async with client_for(app) as http:
        assert (await http.get("/v1/ping", headers=_auth("nope"))).status_code == 401
