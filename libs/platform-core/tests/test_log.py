"""structlog configuration.

Assertions are made against captured stdout rather than a structlog test
double, because the thing worth guarding is the shape of the line an operator
or log shipper actually sees.
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from platform_core.context import request_context
from platform_core.log import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    structlog.contextvars.clear_contextvars()
    yield
    structlog.reset_defaults()


def _emitted(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    out = capsys.readouterr().out.strip()
    return [json.loads(line) for line in out.splitlines()] if out else []


def test_json_line_shape(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    get_logger("kb_api.test").info("document ingested", document_id="7", chunks=12)

    (event,) = _emitted(capsys)
    assert event["event"] == "document ingested"
    assert event["level"] == "info"
    assert event["document_id"] == "7"
    assert event["chunks"] == 12
    assert event["timestamp"].endswith("Z")


def test_level_filters_below_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING")
    logger = get_logger()
    logger.info("noise")
    logger.warning("signal")

    assert [event["event"] for event in _emitted(capsys)] == ["signal"]


def test_level_is_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="debug")
    get_logger().debug("verbose")
    assert _emitted(capsys)[0]["event"] == "verbose"


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level: 'chatty'"):
        configure_logging(level="chatty")


def test_console_output_is_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=False)
    get_logger().info("human readable")

    out = capsys.readouterr().out
    assert "human readable" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_initial_values_are_bound_to_every_event(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    logger = get_logger("kb_api", service="kb-api")
    logger.info("first")
    logger.info("second")

    assert [event["service"] for event in _emitted(capsys)] == ["kb-api", "kb-api"]


def test_request_id_is_merged_from_the_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    with request_context() as request_id:
        get_logger().info("inside a request")

    (event,) = _emitted(capsys)
    assert event["request_id"] == request_id


def test_exception_info_is_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    try:
        raise RuntimeError("upstream refused")
    except RuntimeError:
        get_logger().exception("embedding call failed")

    (event,) = _emitted(capsys)
    assert "RuntimeError: upstream refused" in event["exception"]


def test_reconfiguring_replaces_the_previous_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level="ERROR")
    configure_logging(level="INFO")
    get_logger().info("visible after reconfigure")
    assert len(_emitted(capsys)) == 1
