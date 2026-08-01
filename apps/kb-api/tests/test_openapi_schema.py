"""The committed contract (AD-016).

Two jobs. The drift check is the one CI runs — it fails when a route changes and
the schema was not re-exported, which is the whole reason the file is committed
rather than generated on demand. The rest assert the properties that make the
document worth reading: that the endpoints exist, that `/health` is open and
everything else is not, and that the bodies carry examples.

The Phase 8 exit criterion is that Swagger UI is "complete enough to drive the
API without reading the source". That is not directly assertable, but its
components are: a summary on every operation, and an example on every request
and response body a caller has to construct.
"""

import json
from http import HTTPStatus
from typing import Any

import pytest

from kb_api.scripts.export_openapi import OPENAPI_PATH, export, schema

Schema = dict[str, Any]
"""`Any` because an OpenAPI document is nested JSON of no fixed shape."""


@pytest.fixture(scope="module")
def document() -> Schema:
    return schema()


def test_the_committed_schema_matches_the_routes() -> None:
    """`just schema` was not run, or its result was not committed."""
    assert export(check=True) == 0, "run `just schema` and commit apps/kb-api/openapi.json"


def test_the_committed_file_is_valid_json_and_openapi_3_1() -> None:
    committed = json.loads(OPENAPI_PATH.read_text())

    assert committed["openapi"].startswith("3.1")
    assert committed["info"]["title"] == "kb-api"


def test_every_prd_endpoint_is_present(document: Schema) -> None:
    paths = document["paths"]

    assert set(paths) == {
        "/health",
        "/v1/search",
        "/v1/documents",
        "/v1/documents/{document_id}",
        "/v1/admin/stats",
    }
    assert set(paths["/v1/documents"]) == {"get", "post"}
    assert set(paths["/v1/documents/{document_id}"]) == {"get", "delete"}


def test_health_is_the_only_open_route(document: Schema) -> None:
    unsecured = [
        f"{method.upper()} {path}"
        for path, operations in document["paths"].items()
        for method, operation in operations.items()
        if "security" not in operation
    ]

    assert unsecured == ["GET /health"]


def test_every_operation_has_a_summary(document: Schema) -> None:
    missing = [
        f"{method.upper()} {path}"
        for path, operations in document["paths"].items()
        for method, operation in operations.items()
        if not operation.get("summary")
    ]

    assert missing == []


@pytest.mark.parametrize(
    "model",
    [
        "IngestDocumentRequest",
        "IngestDocumentResponse",
        "SearchRequest",
        "SearchResponse",
        "DocumentListResponse",
        "StatsResponse",
    ],
)
def test_the_bodies_a_caller_constructs_carry_examples(document: Schema, model: str) -> None:
    """Examples are most of what makes Swagger UI drivable without the source."""
    assert document["components"]["schemas"][model].get("examples")


def test_the_scoped_routes_document_their_403(document: Schema) -> None:
    """A `403` that is not in the schema reads as a bug when a caller hits it."""
    paths = document["paths"]
    scoped = [
        paths["/v1/documents"]["post"],
        paths["/v1/documents"]["get"],
        paths["/v1/documents/{document_id}"]["delete"],
        paths["/v1/admin/stats"]["get"],
        paths["/v1/search"]["post"],
    ]

    for operation in scoped:
        assert str(HTTPStatus.FORBIDDEN.value) in operation["responses"], operation["summary"]


def test_rate_limiting_is_documented_on_every_v1_route(document: Schema) -> None:
    """AD-014 rejects with a `429` on any `/v1` path; a client that does not know
    to back off will hammer straight through the limit."""
    for path, operations in document["paths"].items():
        if not path.startswith("/v1"):
            continue
        for method, operation in operations.items():
            assert str(HTTPStatus.TOO_MANY_REQUESTS.value) in operation["responses"], (
                f"{method.upper()} {path}"
            )
