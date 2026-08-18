"""Shared settings, logging, and error primitives for Redshift7 services."""

from platform_core.auth import (
    ApiKeyRegistry,
    ApiKeys,
    ApiKeyScopes,
    Principal,
)
from platform_core.context import (
    REQUEST_ID_HEADER,
    coerce_request_id,
    get_request_id,
    new_request_id,
    request_context,
)
from platform_core.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    PlatformError,
    RateLimitedError,
    UpstreamError,
    ValidationError,
)
from platform_core.log import configure_logging, get_logger
from platform_core.settings import BaseServiceSettings, Environment, LogLevel

__all__ = [
    "REQUEST_ID_HEADER",
    "ApiKeyRegistry",
    "ApiKeyScopes",
    "ApiKeys",
    "AuthenticationError",
    "AuthorizationError",
    "BaseServiceSettings",
    "ConfigurationError",
    "ConflictError",
    "Environment",
    "LogLevel",
    "NotFoundError",
    "PayloadTooLargeError",
    "PlatformError",
    "Principal",
    "RateLimitedError",
    "UpstreamError",
    "ValidationError",
    "__version__",
    "coerce_request_id",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "request_context",
]

__version__ = "0.1.0"
