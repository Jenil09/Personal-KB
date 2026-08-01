"""The error hierarchy every service raises.

Services and adapters raise these; the single exception handler in
`platform-fastapi` is the only place that turns one into an HTTP response
(RFC 9457 problem+json). Nothing outside that handler constructs
`HTTPException` — see ai-kb/TECHNICAL-DESIGN.md §5.

`code` is part of the API contract. Clients branch on it and the tier-1 audit
trail stores it, so renaming one is a breaking change; add a new subclass
instead.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ConflictError",
    "NotFoundError",
    "PayloadTooLargeError",
    "PlatformError",
    "RateLimitedError",
    "UpstreamError",
    "ValidationError",
]


class PlatformError(Exception):
    """Base for every expected failure with a defined HTTP mapping."""

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"
    title: ClassVar[str] = "Internal Server Error"

    def __init__(
        self,
        detail: str,
        *,
        context: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        # `Any` because context holds arbitrary JSON-serialisable extension
        # members destined for the problem+json body and the audit row.
        super().__init__(detail)
        self.detail = detail
        self.context: dict[str, Any] = dict(context) if context else {}
        # `Retry-After` on a 429 is the case this exists for: it is part of the
        # answer rather than decoration, and a client that has to parse the body
        # to find it will not.
        self.headers: dict[str, str] = dict(headers) if headers else {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"status_code={self.status_code}, detail={self.detail!r})"
        )


class NotFoundError(PlatformError):
    status_code: ClassVar[int] = 404
    code: ClassVar[str] = "not_found"
    title: ClassVar[str] = "Not Found"


class AuthenticationError(PlatformError):
    """No usable API key was presented (AD-011).

    Deliberately undifferentiated: a missing key, a malformed header, and an
    unrecognised key all answer `401` with the same code, so the response
    cannot be used to probe which keys exist. The distinction is in `detail`
    and in the audit row, not in the contract.
    """

    status_code: ClassVar[int] = 401
    code: ClassVar[str] = "unauthenticated"
    title: ClassVar[str] = "Unauthorized"


class AuthorizationError(PlatformError):
    """A recognised key that lacks the scope the operation needs (AD-024).

    Differentiated where `AuthenticationError` deliberately is not, and the
    asymmetry is the point. A `401` must not reveal whether a key exists, so it
    says nothing. A caller reaching here already holds a valid key and already
    knows it exists — telling it which scope it is missing is diagnosable rather
    than leaky, so `context` carries the required scope.
    """

    status_code: ClassVar[int] = 403
    code: ClassVar[str] = "insufficient_scope"
    title: ClassVar[str] = "Forbidden"


class RateLimitedError(PlatformError):
    """Too many requests from one key (AD-014).

    Raised by the two limits that reject — the per-minute burst ceiling and the
    daily hard ceiling. The middle threshold flags the audit row and does not
    raise, because a limit that blocks legitimate traffic on a productive day is
    worse than one that only tells you the day was unusual.
    """

    status_code: ClassVar[int] = 429
    code: ClassVar[str] = "rate_limited"
    title: ClassVar[str] = "Too Many Requests"


class ValidationError(PlatformError):
    """A request or payload that parsed but is not acceptable."""

    status_code: ClassVar[int] = 422
    code: ClassVar[str] = "validation_error"
    title: ClassVar[str] = "Unprocessable Entity"


class PayloadTooLargeError(PlatformError):
    """The request body is over the configured ceiling (Design §8).

    Raised by the body-cap middleware before the body is buffered, so an
    oversized request costs the bytes already on the wire and nothing else.
    """

    status_code: ClassVar[int] = 413
    code: ClassVar[str] = "payload_too_large"
    title: ClassVar[str] = "Content Too Large"


class ConflictError(PlatformError):
    """State prevents the operation — a duplicate, or an unpopulated collection."""

    status_code: ClassVar[int] = 409
    code: ClassVar[str] = "conflict"
    title: ClassVar[str] = "Conflict"


class UpstreamError(PlatformError):
    """A dependency we do not control failed: embedding provider, Chroma, Postgres."""

    status_code: ClassVar[int] = 502
    code: ClassVar[str] = "upstream_error"
    title: ClassVar[str] = "Bad Gateway"


class ConfigurationError(PlatformError):
    """Misconfiguration. Raised at startup, never mid-request."""

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "configuration_error"
    title: ClassVar[str] = "Internal Server Error"
