"""Request-ID generation and propagation."""

from uuid import UUID

import structlog

from platform_core.context import (
    REQUEST_ID_HEADER,
    coerce_request_id,
    get_request_id,
    new_request_id,
    request_context,
)


def test_header_name() -> None:
    assert REQUEST_ID_HEADER == "X-Request-ID"


def test_new_request_id_is_a_unique_uuid() -> None:
    first, second = new_request_id(), new_request_id()
    assert UUID(first) and UUID(second)
    assert first != second


def test_coerce_keeps_a_valid_uuid_and_canonicalises_it() -> None:
    supplied = "0f2a4c8e-1b3d-4f5a-8c7b-9d0e1f2a3b4c"
    assert coerce_request_id(supplied) == supplied
    assert coerce_request_id(f"  {supplied.upper()}  ") == supplied


def test_coerce_replaces_anything_that_is_not_a_uuid() -> None:
    """A Postgres UUID column is downstream of this, so junk cannot propagate."""
    for junk in (None, "", "not-a-uuid", "'; DROP TABLE kb.documents; --"):
        assert UUID(coerce_request_id(junk))


def test_no_request_id_outside_a_context() -> None:
    assert get_request_id() is None


def test_context_binds_and_restores() -> None:
    supplied = "0f2a4c8e-1b3d-4f5a-8c7b-9d0e1f2a3b4c"
    with request_context(supplied) as request_id:
        assert request_id == supplied
        assert get_request_id() == supplied
    assert get_request_id() is None


def test_context_generates_when_the_caller_supplies_nothing() -> None:
    with request_context() as request_id:
        assert get_request_id() == request_id
        assert UUID(request_id)


def test_nested_contexts_restore_the_outer_id() -> None:
    outer = "0f2a4c8e-1b3d-4f5a-8c7b-9d0e1f2a3b4c"
    inner = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
    with request_context(outer):
        with request_context(inner):
            assert get_request_id() == inner
        assert get_request_id() == outer


def test_request_id_reaches_structlog_contextvars() -> None:
    structlog.contextvars.clear_contextvars()
    with request_context() as request_id:
        assert structlog.contextvars.get_contextvars()["request_id"] == request_id
    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_context_unwinds_on_exception() -> None:
    try:
        with request_context():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert get_request_id() is None
