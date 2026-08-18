"""Fixtures for the `kb-mcp` suite.

The `/v1` fake is `kb_client.testing.FakeService` and nothing here reimplements
it. **Nothing imports `kb_api` either**, which is the rule AD-025 set for
`kb-cli` and it holds for the same reason: this server's claim is that it speaks
the published contract and can be older than the service it talks to. A test that
imported the service would assert the two agree on this commit — which they
trivially do, and which proves nothing about a deployed pair.

`_isolated_env` is autouse because these settings read the environment and a
`.env` file. Without it, a developer with `KB_MCP__*` exported — or with the
repository's own `.env` in the working directory — would get different results
from CI.
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from kb_client.client import KbClient
from kb_client.testing import API_KEY, FakeService
from kb_mcp.config import KbMcpSettings

HOST_TOKEN = "host-token"
"""The bearer token an MCP host presents to `kb-mcp`.

Deliberately not `API_KEY`: the token a host presents here and the key this
server presents to `kb-api` are two different credentials, and a suite that used
one string for both could not notice them being confused.
"""

READ_ONLY_TOKEN = "read-only-token"
WRITE_ONLY_TOKEN = "write-only-token"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in [n for n in os.environ if n.startswith(("KB_MCP__", "KB_CLIENT__"))]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def settings() -> KbMcpSettings:
    """Three keys, covering all three states a key can be in.

    `claude` is named in no scope entry and is therefore unrestricted — AD-024's
    permissive default, and the state an operator lands in by configuring keys
    and forgetting scopes. The other two prove the split is real in both
    directions: a system that only ever refuses writes has not been shown to
    refuse anything.
    """
    return KbMcpSettings(
        kb_api={"base_url": "http://kb.test", "api_key": API_KEY},
        api_keys={"claude": HOST_TOKEN, "reader": READ_ONLY_TOKEN, "writer": WRITE_ONLY_TOKEN},
        api_key_scopes={"reader": frozenset({"search"}), "writer": frozenset({"write"})},
    )


@pytest.fixture
async def client(settings: KbMcpSettings, service: FakeService) -> AsyncIterator[KbClient]:
    kb_client = KbClient(settings.kb_api, transport=service.transport)
    try:
        yield kb_client
    finally:
        await kb_client.aclose()
