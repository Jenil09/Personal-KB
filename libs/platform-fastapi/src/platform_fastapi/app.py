"""The app factory every service's composition root calls.

`create_app` is the only place that knows how the cross-cutting pieces fit
together: logging configured before anything logs, the key registry built once,
error handlers installed, middleware ordered, `/health` open and every `/v1`
router closed.

Middleware order matters and is fixed here (outermost first):

    ServerErrorMiddleware   starlette's, always outermost
    CORSMiddleware          so its headers reach error responses too
    RequestContextMiddleware
    AuditMiddleware         inside the context, so the row carries the request ID
    BodySizeLimitMiddleware inside the audit, so its 413 is a recorded request
    RateLimitMiddleware     inside the audit, so its 429 is a recorded request
    ExceptionMiddleware     starlette's, runs the handlers below
    router

The audit layer's position is the whole of AD-013's coverage guarantee. Every
rejection below it — oversized body, throttled key, unrecognised credential,
unknown path — is a response the trail has to hold, and each one is produced by
a layer the audit middleware wraps. Moving it inward by one would lose a class
of request, and each class it would lose is one the trail exists for.

`add_middleware` prepends, so the calls below read in the opposite order.
"""

from collections.abc import Callable, Sequence
from typing import Any

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import Lifespan

from platform_core import REQUEST_ID_HEADER, configure_logging, get_logger
from platform_db import AuditRecord, AuditTrail
from platform_fastapi.audit import AuditMiddleware
from platform_fastapi.auth import ApiKeyRegistry, require_api_key
from platform_fastapi.body_limit import BodySizeLimitMiddleware
from platform_fastapi.health import HealthCheck, create_health_router
from platform_fastapi.middleware import RequestContextMiddleware
from platform_fastapi.problem import install_error_handlers
from platform_fastapi.ratelimit import (
    BurstDetector,
    RateLimiter,
    RateLimitMiddleware,
    RateLimitSettings,
)
from platform_fastapi.settings import HttpServiceSettings

__all__ = ["API_PREFIX", "create_app"]

_logger = get_logger("platform.startup")

API_PREFIX = "/v1"
"""Every route is versioned; `/health` is not (AD-012)."""


def create_app(
    settings: HttpServiceSettings,
    *,
    routers: Sequence[APIRouter] = (),
    lifespan: Lifespan[Any] | None = None,
    health_checks: Sequence[HealthCheck] = (),
    audit_trail: AuditTrail | None = None,
    audit_observer: Callable[[AuditRecord, BaseException | None], None] | None = None,
    rate_limits: RateLimitSettings | None = None,
) -> FastAPI:
    """Assemble a service. `routers` are mounted under `/v1` behind auth.

    `audit_trail` is optional so a service can be assembled without a database —
    the ten-line example in the Phase 2 tests, and every unit test that only
    wants a router. It is not optional in a deployment: `kb-api` passes one, and
    without it AD-013's guarantee is simply absent rather than degraded.
    """
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    registry = ApiKeyRegistry(settings.api_keys, settings.api_key_scopes)
    app.state.api_key_registry = registry

    # Logged, not merely configured. A key with no scope entry is unrestricted
    # (AD-024), which is the permissive direction — so the grants are stated at
    # startup where an omission is caught by reading the log, rather than
    # discovered later by an incident.
    _logger.info(
        "api_keys_loaded",
        grants={
            key_id: sorted(scopes) if scopes is not None else "unrestricted"
            for key_id, scopes in registry.grants
        },
    )

    install_error_handlers(app)

    limits = rate_limits or RateLimitSettings()
    app.add_middleware(
        RateLimitMiddleware,
        limiter=RateLimiter(limits),
        registry=registry,
        # `/health` is exempt for the same reason it is unauthenticated: it is
        # polled by the orchestrator on a fixed schedule, and throttling it turns
        # a busy minute into a restart.
        exempt_paths=("/health",),
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    if audit_trail is not None:
        app.add_middleware(
            AuditMiddleware,
            trail=audit_trail,
            bursts=BurstDetector(limits),
            observer=audit_observer,
        )
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER],
        )

    app.include_router(
        create_health_router(
            version=settings.service_version,
            checks=health_checks,
            timeout_seconds=settings.health_timeout_seconds,
        )
    )
    for router in routers:
        app.include_router(router, prefix=API_PREFIX, dependencies=[Depends(require_api_key)])

    return app
