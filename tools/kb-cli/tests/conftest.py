"""Fixtures for the `kb-cli` suite.

The `/v1` contract fake is not here — it ships as `kb_client.testing.FakeService`
so that `kb-cli`, `kb-client`'s own suite, and `kb-mcp` all assert against one
copy of the contract. What is left is the half that belongs to this tool: a fake
for the metadata suggester's model (AD-026), and the isolation that keeps a
developer's real configuration out of the tests.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from kb_cli.config import CONFIG_PATH_ENV_VAR, KbCliSettings, SuggestSettings
from kb_client.testing import API_KEY, FakeService

MODEL_URL = "http://model.test/v1"


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

    A fixture as well as an importable constant, because most of what the suite
    shares still has to cross as a fixture: `tests/` has no `__init__.py`
    (deliberately — see CLAUDE.md), so a test module cannot import from
    `conftest` in a way mypy can also resolve. That is why the suite's
    parameters are unannotated and why `no-untyped-def` is off for test suites.
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
