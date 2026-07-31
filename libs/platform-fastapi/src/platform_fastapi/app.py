"""The app factory every service's composition root calls.

`create_app` is the only place that knows how the cross-cutting pieces fit
together: logging configured before anything logs, the key registry built once,
error handlers installed, middleware ordered, `/health` open and every `/v1`
router closed.

Middleware order matters and is fixed here (outermost first):

    ServerErrorMiddleware   starlette's, always outermost
    CORSMiddleware          so its headers reach error responses too
    RequestContextMiddleware
    ExceptionMiddleware     starlette's, runs the handlers below
    router

`add_middleware` prepends, so the calls below read in the opposite order.
"""

from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import Lifespan

from platform_core import REQUEST_ID_HEADER, configure_logging
from platform_fastapi.auth import ApiKeyRegistry, require_api_key
from platform_fastapi.health import HealthCheck, create_health_router
from platform_fastapi.middleware import RequestContextMiddleware
from platform_fastapi.problem import install_error_handlers
from platform_fastapi.settings import HttpServiceSettings

__all__ = ["API_PREFIX", "create_app"]

API_PREFIX = "/v1"
"""Every route is versioned; `/health` is not (AD-012)."""


def create_app(
    settings: HttpServiceSettings,
    *,
    routers: Sequence[APIRouter] = (),
    lifespan: Lifespan[Any] | None = None,
    health_checks: Sequence[HealthCheck] = (),
) -> FastAPI:
    """Assemble a service. `routers` are mounted under `/v1` behind auth."""
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.api_key_registry = ApiKeyRegistry(settings.api_keys)

    install_error_handlers(app)

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
