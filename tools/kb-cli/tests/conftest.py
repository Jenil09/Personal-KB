"""A stand-in for `kb-api` that speaks the `/v1` contract.

`httpx.MockTransport`, not the real app. `kb-cli` does not depend on `kb-api`
(AD-025) and must not start doing so in its tests: the client's whole claim is
that it parses *the contract* and can therefore be older than the service it
talks to, and a test that imported the service would instead assert that the two
agree on this commit — which they trivially do, and which proves nothing.

So the interesting parts are reproduced faithfully rather than conveniently:

* errors are RFC 9457 problem+json with the `code` extension member, because
  that is what the client maps back into the `PlatformError` hierarchy
* `DELETE` answers `204` whether or not the document existed, and says which in
  `X-Deleted` (PRD §6.6)
* listing pages on `limit`/`offset` and reports `total` for the whole match, so
  the "last page is exactly full" case is reachable
* re-ingesting identical content answers `unchanged` with zero chunks created
  (AD-008), and ingesting over an existing `source` reports `superseded`
  (AD-020)
"""

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from kb_cli.config import CONFIG_PATH_ENV_VAR, KbCliSettings, SuggestSettings

API_KEY = "test-operator-key"
MODEL_URL = "http://model.test/v1"
COLLECTION = "kb__openai__text_embedding_3_small__1536__c1"


class FakeService:
    """Mutable corpus state plus a request handler over it."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        self.search_hits: list[dict[str, Any]] = []
        self.fail_next: tuple[int, str, str] | None = None

    # --- seeding ----------------------------------------------------------

    def add(
        self,
        title: str,
        *,
        content: str = "# Body\n\nSome text.",
        type: str = "note",  # noqa: A002 — the wire name (PRD §6.4)
        tags: tuple[str, ...] = (),
        source: str | None = None,
        document_id: str | None = None,
    ) -> str:
        identifier = document_id or str(uuid.uuid4())
        stamp = datetime.now(UTC).isoformat()
        self.documents[identifier] = {
            "id": identifier,
            "title": title,
            "type": type,
            "collection": COLLECTION,
            "status": "indexed",
            "chunk_count": max(1, len(content) // 40),
            "content_hash": f"{abs(hash(content)):x}",
            "created_at": stamp,
            "updated_at": stamp,
            "source": source,
            "tags": list(tags),
            "content": content,
            "ingested_by_key_id": "cli",
            "ingested_from_ip": "127.0.0.1",
        }
        return identifier

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # --- routing ----------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        if request.headers.get("Authorization") != f"Bearer {API_KEY}":
            return self._problem(request, 401, "unauthenticated", "API key is not recognised.")

        if self.fail_next is not None:
            status, code, detail = self.fail_next
            self.fail_next = None
            return self._problem(request, status, code, detail)

        if path == "/v1/documents" and request.method == "GET":
            return self._list(request)
        if path == "/v1/documents" and request.method == "POST":
            return self._ingest(request)
        if path == "/v1/search" and request.method == "POST":
            return self._search(request)
        if path == "/v1/admin/stats":
            return httpx.Response(200, json=self._stats())
        if path.startswith("/v1/documents/"):
            identifier = path.rsplit("/", 1)[-1]
            if request.method == "GET":
                return self._get(request, identifier)
            if request.method == "DELETE":
                return self._delete(identifier)

        return self._problem(request, 404, "not_found", f"No route for {path}.")

    # --- handlers ---------------------------------------------------------

    def _list(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        matched = list(self.documents.values())
        if wanted := params.get("type"):
            matched = [d for d in matched if d["type"] == wanted]
        if source := params.get("source"):
            matched = [d for d in matched if d["source"] == source]
        tags = params.get_list("tags")
        if tags:
            if params.get("match_all_tags") in ("true", "True"):
                matched = [d for d in matched if set(tags) <= set(d["tags"])]
            else:
                matched = [d for d in matched if set(tags) & set(d["tags"])]
        limit = int(params.get("limit", 50))
        offset = int(params.get("offset", 0))
        page = matched[offset : offset + limit]
        return httpx.Response(
            200,
            json={
                "documents": [_summary(d) for d in page],
                "total": len(matched),
                "limit": limit,
                "offset": offset,
            },
        )

    def _get(self, request: httpx.Request, identifier: str) -> httpx.Response:
        document = self.documents.get(identifier)
        if document is None:
            return self._problem(request, 404, "not_found", f"No document with id {identifier}.")
        return httpx.Response(200, json=document)

    def _delete(self, identifier: str) -> httpx.Response:
        existed = self.documents.pop(identifier, None) is not None
        # 204 either way; the header carries which (PRD §6.6).
        return httpx.Response(204, headers={"X-Deleted": "true" if existed else "false"})

    def _ingest(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        source = body.get("source")
        superseded = [
            identifier
            for identifier, document in self.documents.items()
            if source is not None and document["source"] == source
        ]
        existing = next(
            (
                document
                for document in self.documents.values()
                if document["content"] == body["content"] and document["title"] == body["title"]
            ),
            None,
        )
        if existing is not None:
            return httpx.Response(
                201,
                json={
                    "document_id": existing["id"],
                    "chunks_created": 0,
                    "chunks_reused": existing["chunk_count"],
                    "total_tokens": 0,
                    "status": "unchanged",
                    "collection": COLLECTION,
                    "superseded": [],
                },
            )
        for identifier in superseded:
            del self.documents[identifier]
        identifier = self.add(
            body["title"],
            content=body["content"],
            type=body["type"],
            tags=tuple(body.get("tags") or ()),
            source=source,
        )
        return httpx.Response(
            201,
            json={
                "document_id": identifier,
                "chunks_created": self.documents[identifier]["chunk_count"],
                "chunks_reused": 0,
                "total_tokens": len(body["content"]) // 4,
                "status": "success",
                "collection": COLLECTION,
                "superseded": superseded,
            },
        )

    def _search(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        hits = self.search_hits or [
            {
                "id": f"{document['id']}:0",
                "text": document["content"],
                "metadata": {
                    "document_id": document["id"],
                    "title": document["title"],
                    "type": document["type"],
                    "tags": document["tags"],
                    "source": document["source"],
                    "ordinal": 0,
                },
                "score": 0.9 - index * 0.1,
            }
            for index, document in enumerate(self.documents.values())
        ]
        return httpx.Response(
            200,
            json={
                "results": hits[: body.get("top_k", 5)],
                "query_tokens": max(1, len(body["query"]) // 4),
                "latency_ms": 42,
            },
        )

    def _stats(self) -> dict[str, Any]:
        return {
            "documents_by_status": {"indexed": len(self.documents)},
            "documents_by_collection": {COLLECTION: len(self.documents)},
            "total_documents": len(self.documents),
            "total_chunks": sum(d["chunk_count"] for d in self.documents.values()),
            "total_tokens_stored": 12345,
            "collections": [
                {
                    "name": COLLECTION,
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "dimensions": 1536,
                    "vectors": sum(d["chunk_count"] for d in self.documents.values()),
                }
            ],
            "tokens": {
                "exact_tokens": 12345,
                "estimated_tokens": 0,
                "api_calls": 7,
                "window_days": 30,
            },
            "telemetry_dropped": 0,
            "telemetry_written": 100,
            "telemetry_queue_depth": 0,
            "audit_spill_depth": 0,
            "recent_bursts": 0,
        }

    @staticmethod
    def _problem(request: httpx.Request, status: int, code: str, detail: str) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "application/problem+json"},
            json={
                "type": "about:blank",
                "title": "Error",
                "status": status,
                "detail": detail,
                "instance": request.url.path,
                "code": code,
                "request_id": str(uuid.uuid4()),
            },
        )


def _summary(document: dict[str, Any]) -> dict[str, Any]:
    """A listing row — everything but the content and the provenance."""
    return {
        key: value
        for key, value in document.items()
        if key not in {"content", "ingested_by_key_id", "ingested_from_ip"}
    }


class FakeModel:
    """A stand-in for an OpenAI-compatible `/chat/completions` endpoint.

    Deliberately not a stand-in for OpenAI specifically. The whole claim of
    `suggest.py` is that one request shape reaches OpenAI, Gemini's
    compatibility endpoint, and LM Studio, so what is reproduced here is the
    parts they agree on — plus the two ways a local runtime differs, both of
    which the suggester has to survive:

    * `reject_json_schema` answers 400 for a `response_format` it does not
      implement, which is what llama.cpp-backed runtimes do
    * `reply` may be set to raw text, so a model that wraps its JSON in a fenced
      block or a sentence of preamble is a case the tests can express
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.answer: dict[str, Any] = {
            "title": "Redshift Architecture",
            "type": "architecture",
            "tags": ["kubernetes", "observability"],
        }
        self.reply: str | None = None
        self.status = 200
        self.error: dict[str, Any] | None = None
        self.reject_json_schema = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.reject_json_schema and body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "'response_format.json_schema' is not supported"}}
            )
        if self.status >= 400:
            return httpx.Response(self.status, json=self.error or {"error": {"message": "nope"}})
        text = self.reply if self.reply is not None else json.dumps(self.answer)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            },
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport)


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def suggest_settings() -> SuggestSettings:
    return SuggestSettings(base_url=MODEL_URL, model="gpt-4o-mini")


@pytest.fixture
def api_key() -> str:
    """The key the fake service accepts.

    A fixture rather than an importable constant: `tests/` has no `__init__.py`
    (deliberately — see CLAUDE.md), so a test module cannot import from
    `conftest` in a way mypy can also resolve. Everything shared crosses the
    boundary as a fixture instead, which is why the suite's parameters are
    unannotated and why `no-untyped-def` is off for test suites.
    """
    return API_KEY


@pytest.fixture
def settings(isolated_config: None) -> KbCliSettings:
    return KbCliSettings(base_url="http://kb.test", api_key=API_KEY)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point every test at a config file that does not exist.

    Autouse and unconditional: without it, whichever machine runs the suite
    contributes its own `~/.config/kb-cli/config.json` to the settings a test
    constructs, and a developer with a real API key stored would see different
    results from CI. The environment is cleared for the same reason.
    """
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(tmp_path / "kb-cli" / "config.json"))
    for variable in (
        "KB_CLI__BASE_URL",
        "KB_CLI__API_KEY",
        "KB_CLI__PROVIDER",
        "KB_CLI__TIMEOUT_SECONDS",
        "KB_CLI__INGEST_TIMEOUT_SECONDS",
        "KB_CLI__EDITOR",
        "KB_CLI__SUGGEST__BASE_URL",
        "KB_CLI__SUGGEST__MODEL",
        "KB_CLI__SUGGEST__API_KEY",
        "KB_CLI__SUGGEST__TIMEOUT_SECONDS",
        "KB_CLI__SUGGEST__MAX_CONTENT_CHARS",
    ):
        monkeypatch.delenv(variable, raising=False)
    yield
