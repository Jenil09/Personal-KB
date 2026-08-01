"""Tier-1 audit coverage (AD-013).

The criterion this suite exists to prove is a counting one — requests in, rows
out, equal — so almost every test here asserts on the number of records rather
than on the contents of one. A trail that logs the happy path and drops the
rejections is the failure mode worth catching, and it passes any spot check.

`RecordingTrail` in `conftest.py` subclasses the real `AuditTrail` and replaces
only its terminal write, so the `AuditRecord` under assertion is the one the
middleware actually constructed and handed over.
"""

from typing import Any
from uuid import UUID

import pytest
from fastapi import APIRouter, Request

from platform_core import REQUEST_ID_HEADER, ConflictError
from platform_db import AuditRecord, AuditTrail, Outcome
from platform_fastapi import RateLimitSettings, record_operation


class RecordingTrail(AuditTrail):
    """An `AuditTrail` whose insert lands in a list instead of Postgres."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)

    @property
    def only(self) -> AuditRecord:
        assert len(self.records) == 1, f"expected exactly one record, got {len(self.records)}"
        return self.records[0]


@pytest.fixture
def trail() -> RecordingTrail:
    return RecordingTrail()


@pytest.fixture
def router() -> APIRouter:
    api = APIRouter()

    @api.post("/search")
    async def search(body: dict[str, Any], request: Request) -> dict[str, bool]:
        record_operation(request, "search", {"query": body.get("query")})
        return {"ok": True}

    @api.get("/boom")
    async def boom() -> dict[str, bool]:
        raise RuntimeError("unhandled")

    @api.get("/conflict")
    async def conflict() -> dict[str, bool]:
        raise ConflictError("nothing to search")

    return api


@pytest.fixture
def client(make_app, client_for, router, trail):
    return client_for(make_app(routers=[router], audit_trail=trail))


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def test_a_successful_request_writes_one_row(client, trail, valid_key) -> None:
    async with client as http:
        response = await http.post("/v1/search", json={"query": "cilium"}, headers=_auth(valid_key))

    assert response.status_code == 200
    record = trail.only
    assert record.key_id == "n8n"
    assert record.method == "POST"
    assert record.path == "/v1/search"
    assert record.status_code == 200
    assert record.outcome is Outcome.SUCCESS
    assert record.operation == "search"
    assert record.payload == {"query": "cilium"}
    assert record.error_code is None


async def test_the_row_carries_the_request_id_the_caller_was_given(
    client, trail, valid_key
) -> None:
    async with client as http:
        response = await http.post("/v1/search", json={"query": "x"}, headers=_auth(valid_key))

    assert trail.only.request_id == UUID(response.headers[REQUEST_ID_HEADER])


async def test_a_rejected_key_is_still_recorded(client, trail) -> None:
    """The case a dependency-based audit would miss entirely."""
    async with client as http:
        response = await http.post("/v1/search", json={"query": "x"}, headers=_auth("wrong"))

    assert response.status_code == 401
    record = trail.only
    assert record.key_id is None
    assert record.outcome is Outcome.AUTH_FAILED
    assert record.error_code == "unauthenticated"
    assert record.operation == "auth_failure"


async def test_a_rejected_key_is_fingerprinted_never_stored(client, trail) -> None:
    async with client as http:
        await http.post("/v1/search", json={"query": "x"}, headers=_auth("hunter2"))

    payload = trail.only.payload
    assert payload is not None
    assert "hunter2" not in str(payload)
    assert len(payload["key_fingerprint"]) == 8


async def test_a_missing_credential_records_no_fingerprint(client, trail) -> None:
    async with client as http:
        await http.post("/v1/search", json={"query": "x"})

    assert trail.only.payload is None


async def test_an_unknown_path_is_recorded(client, trail, valid_key) -> None:
    async with client as http:
        response = await http.get("/v1/nothing-here", headers=_auth(valid_key))

    assert response.status_code == 404
    assert trail.only.path == "/v1/nothing-here"
    assert trail.only.outcome is Outcome.CLIENT_ERROR


async def test_a_platform_error_records_its_code(client, trail, valid_key) -> None:
    async with client as http:
        response = await http.get("/v1/conflict", headers=_auth(valid_key))

    assert response.status_code == 409
    assert trail.only.error_code == "conflict"
    assert trail.only.outcome is Outcome.CLIENT_ERROR


async def test_an_unhandled_exception_is_recorded_and_still_answered(
    client, trail, valid_key
) -> None:
    """Recorded, then re-raised: the row must not describe a response the caller
    never received, and the caller must still get their 500."""
    async with client as http:
        response = await http.get("/v1/boom", headers=_auth(valid_key))

    assert response.status_code == 500
    record = trail.only
    assert record.status_code == 500
    assert record.outcome is Outcome.SERVER_ERROR
    assert record.error_code == "internal_error"


async def test_an_oversized_body_is_recorded(
    make_app, client_for, router, trail, valid_key
) -> None:
    app = make_app(routers=[router], audit_trail=trail, max_body_bytes=32)
    async with client_for(app) as http:
        response = await http.post(
            "/v1/search", json={"query": "x" * 200}, headers=_auth(valid_key)
        )

    assert response.status_code == 413
    assert trail.only.error_code == "payload_too_large"


async def test_health_is_audited_too(client, trail) -> None:
    """Unauthenticated does not mean unrecorded. AD-013 says every request."""
    async with client as http:
        assert (await http.get("/health")).status_code == 200

    assert trail.only.path == "/health"
    assert trail.only.key_id is None


async def test_every_request_produces_exactly_one_row(client, trail, valid_key) -> None:
    """The Phase 8 exit criterion, by count rather than by spot check."""
    async with client as http:
        for index in range(12):
            await http.post("/v1/search", json={"query": f"q{index}"}, headers=_auth(valid_key))
        for _ in range(3):
            await http.post("/v1/search", json={"query": "x"}, headers=_auth("wrong"))
        await http.get("/v1/nothing", headers=_auth(valid_key))
        await http.get("/v1/boom", headers=_auth(valid_key))
        await http.get("/health")

    assert len(trail.records) == 18
    assert len({record.request_id for record in trail.records}) == 18


async def test_latency_is_recorded_as_a_non_negative_integer(client, trail, valid_key) -> None:
    async with client as http:
        await http.post("/v1/search", json={"query": "x"}, headers=_auth(valid_key))

    assert trail.only.latency_ms >= 0


async def test_the_peer_address_is_recorded_when_it_is_one(client, trail, valid_key) -> None:
    """The column is `INET`. A transport reporting a non-address host — some ASGI
    transports use a literal `"testclient"` — must record `None` rather than a
    made-up value, which is what `_client_ip` guards."""
    async with client as http:
        await http.post("/v1/search", json={"query": "x"}, headers=_auth(valid_key))

    assert str(trail.only.client_ip) == "127.0.0.1"


async def test_a_service_without_a_trail_still_serves(make_app, client_for, router, valid_key):
    """The audit layer is optional to assemble and mandatory to deploy."""
    async with client_for(make_app(routers=[router])) as http:
        response = await http.post("/v1/search", json={"query": "x"}, headers=_auth(valid_key))

    assert response.status_code == 200


# --- repeat-burst detection (AD-014) --------------------------------------


async def test_identical_requests_raise_the_repeat_burst_flag(
    make_app, client_for, router, trail, valid_key
) -> None:
    app = make_app(
        routers=[router],
        audit_trail=trail,
        rate_limits=RateLimitSettings(repeat_burst_count=3),
    )
    async with client_for(app) as http:
        for _ in range(4):
            await http.post("/v1/search", json={"query": "same"}, headers=_auth(valid_key))

    assert [record.repeat_burst for record in trail.records] == [False, False, True, True]


async def test_a_busy_run_of_distinct_queries_is_not_a_burst(
    make_app, client_for, router, trail, valid_key
) -> None:
    """n8n's own workload: ten searches on one path within a minute. The payload
    is what separates that from a loop, which is why the fingerprint includes it."""
    app = make_app(
        routers=[router],
        audit_trail=trail,
        rate_limits=RateLimitSettings(repeat_burst_count=3),
    )
    async with client_for(app) as http:
        for index in range(10):
            await http.post("/v1/search", json={"query": f"job-{index}"}, headers=_auth(valid_key))

    assert not any(record.repeat_burst for record in trail.records)
