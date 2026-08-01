"""Shared fixtures.

Every app under test is exercised over `ASGITransport` rather than a live
server: same middleware stack, same handlers, no socket.
"""

from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from pydantic_settings import SettingsConfigDict

from platform_db import AuditRecord, AuditTrail
from platform_fastapi import HealthCheck, HttpServiceSettings, create_app

_KEYS = {"n8n": "n8n-secret-value", "cli": "cli-secret-value"}


class RecordingTrail(AuditTrail):
    """An `AuditTrail` whose terminal write lands in a list instead of Postgres.

    A subclass rather than a mock: the record under assertion is then the one the
    middleware actually constructed and handed over, so a change that builds the
    wrong record cannot pass merely by calling the method.
    """

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


class ExampleSettings(HttpServiceSettings):
    model_config = SettingsConfigDict(env_prefix="EXAMPLE__")


@pytest.fixture
def valid_key() -> str:
    return _KEYS["n8n"]


@pytest.fixture
def other_key() -> str:
    return _KEYS["cli"]


@pytest.fixture
def settings() -> ExampleSettings:
    return ExampleSettings(
        service_name="example",
        service_version="1.2",
        api_keys=dict(_KEYS),
        log_json=True,
    )


@pytest.fixture
def make_app(settings: ExampleSettings):
    """Build an app. Keyword arguments `create_app` accepts are forwarded to it;
    everything else is treated as a settings override."""

    def factory(
        routers: Sequence[APIRouter] = (),
        health_checks: Sequence[HealthCheck] = (),
        **overrides: object,
    ) -> FastAPI:
        passthrough = {
            name: overrides.pop(name)
            for name in ("audit_trail", "rate_limits", "lifespan")
            if name in overrides
        }
        return create_app(
            settings.model_copy(update=overrides) if overrides else settings,
            routers=routers,
            health_checks=health_checks,
            **passthrough,  # type: ignore[arg-type]
        )

    return factory


@pytest.fixture
def client_for():
    """An async client bound to an app, with server exceptions left to the app.

    `raise_app_exceptions=False` is deliberate: the catch-all in
    `RequestContextMiddleware` re-raises nothing, but starlette's own
    `ServerErrorMiddleware` sits outside it and always re-raises, so without
    this the transport would surface the exception instead of the 500 the
    client would really receive.
    """

    def factory(app: FastAPI) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    return factory


@pytest.fixture
async def client(make_app, client_for) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(make_app()) as http_client:
        yield http_client
