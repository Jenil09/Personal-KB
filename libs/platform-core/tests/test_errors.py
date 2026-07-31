"""The error hierarchy is API contract, so the codes and statuses are asserted
literally rather than derived — a rename should fail here loudly."""

import pytest

from platform_core.errors import (
    AuthenticationError,
    ConfigurationError,
    ConflictError,
    NotFoundError,
    PlatformError,
    UpstreamError,
    ValidationError,
)

EXPECTED = [
    (PlatformError, 500, "internal_error", "Internal Server Error"),
    (AuthenticationError, 401, "unauthenticated", "Unauthorized"),
    (NotFoundError, 404, "not_found", "Not Found"),
    (ValidationError, 422, "validation_error", "Unprocessable Entity"),
    (ConflictError, 409, "conflict", "Conflict"),
    (UpstreamError, 502, "upstream_error", "Bad Gateway"),
    (ConfigurationError, 500, "configuration_error", "Internal Server Error"),
]


@pytest.mark.parametrize(("error_cls", "status_code", "code", "title"), EXPECTED)
def test_contract(error_cls: type[PlatformError], status_code: int, code: str, title: str) -> None:
    assert error_cls.status_code == status_code
    assert error_cls.code == code
    assert error_cls.title == title


def test_every_subclass_is_a_platform_error() -> None:
    for error_cls, *_ in EXPECTED:
        assert issubclass(error_cls, PlatformError)


def test_codes_are_unique() -> None:
    codes = [error_cls.code for error_cls, *_ in EXPECTED]
    assert len(set(codes)) == len(codes)


def test_detail_is_the_exception_message() -> None:
    error = NotFoundError("document 7 does not exist")
    assert error.detail == "document 7 does not exist"
    assert str(error) == "document 7 does not exist"


def test_context_defaults_to_empty_and_is_copied() -> None:
    assert PlatformError("boom").context == {}

    supplied = {"provider": "openai"}
    error = UpstreamError("embedding call failed", context=supplied)
    assert error.context == supplied

    supplied["provider"] = "gemini"
    assert error.context == {"provider": "openai"}


def test_repr_names_code_and_status() -> None:
    assert repr(ConflictError("collection is empty")) == (
        "ConflictError(code='conflict', status_code=409, detail='collection is empty')"
    )


def test_is_catchable_as_platform_error() -> None:
    with pytest.raises(PlatformError) as caught:
        raise ValidationError("tags must be scalars")
    assert caught.value.status_code == 422
