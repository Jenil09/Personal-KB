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
    "ConfigurationError",
    "ConflictError",
    "NotFoundError",
    "PlatformError",
    "UpstreamError",
    "ValidationError",
]


class PlatformError(Exception):
    """Base for every expected failure with a defined HTTP mapping."""

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"
    title: ClassVar[str] = "Internal Server Error"

    def __init__(self, detail: str, *, context: Mapping[str, Any] | None = None) -> None:
        # `Any` because context holds arbitrary JSON-serialisable extension
        # members destined for the problem+json body and the audit row.
        super().__init__(detail)
        self.detail = detail
        self.context: dict[str, Any] = dict(context) if context else {}

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


class ValidationError(PlatformError):
    """A request or payload that parsed but is not acceptable."""

    status_code: ClassVar[int] = 422
    code: ClassVar[str] = "validation_error"
    title: ClassVar[str] = "Unprocessable Entity"


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
