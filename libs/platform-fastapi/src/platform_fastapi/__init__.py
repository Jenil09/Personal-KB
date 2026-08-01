"""FastAPI composition: app factory, auth, problem+json errors, health."""

from platform_fastapi.app import API_PREFIX, create_app
from platform_fastapi.audit import AuditMiddleware, record_operation
from platform_fastapi.auth import (
    ApiKeyRegistry,
    CurrentPrincipal,
    Principal,
    require_api_key,
    require_scope,
)
from platform_fastapi.body_limit import BodySizeLimitMiddleware
from platform_fastapi.health import CheckResult, HealthCheck, HealthStatus, create_health_router
from platform_fastapi.middleware import RequestContextMiddleware
from platform_fastapi.problem import (
    PROBLEM_CONTENT_TYPE,
    install_error_handlers,
    problem_response,
)
from platform_fastapi.ratelimit import (
    BurstDetector,
    RateLimiter,
    RateLimitMiddleware,
    RateLimitSettings,
    burst_fingerprint,
)
from platform_fastapi.settings import HttpServiceSettings

__all__ = [
    "API_PREFIX",
    "PROBLEM_CONTENT_TYPE",
    "ApiKeyRegistry",
    "AuditMiddleware",
    "BodySizeLimitMiddleware",
    "BurstDetector",
    "CheckResult",
    "CurrentPrincipal",
    "HealthCheck",
    "HealthStatus",
    "HttpServiceSettings",
    "Principal",
    "RateLimitMiddleware",
    "RateLimitSettings",
    "RateLimiter",
    "RequestContextMiddleware",
    "__version__",
    "burst_fingerprint",
    "create_app",
    "create_health_router",
    "install_error_handlers",
    "problem_response",
    "record_operation",
    "require_api_key",
    "require_scope",
]

__version__ = "0.1.0"
