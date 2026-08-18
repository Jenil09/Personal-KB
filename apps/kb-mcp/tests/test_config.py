"""`KbMcpSettings` — the prefixes, and the one that could have leaked.

The interesting case is not that the fields parse. It is that
`KbApiConnection` overrides `KbClientSettings`' `KB_CLIENT__` prefix, because a
`BaseSettings` subclass used as a field runs its own environment source during
validation: with the inherited prefix, a bare `KB_CLIENT__API_KEY` left in a
shell reaches `KbMcpSettings.kb_api`. It does not override an explicit
`KB_MCP__KB_API__API_KEY` — that wins — but it fills in for a missing one, which
is worse, because the operator who set only `BASE_URL` gets a working server
pointed at their service with a credential they did not mean to give it.

The key format is `platform_core.auth`'s and is tested there in every malformed
shape. What is asserted here is that this class still declares those annotated
types, so a `KB_MCP__API_KEYS` that stopped parsing would fail here too.
"""

import pytest
from pydantic import SecretStr

from kb_mcp.config import KbApiConnection, KbMcpSettings
from platform_core import ConfigurationError


def _minimal() -> KbMcpSettings:
    return KbMcpSettings(
        kb_api=KbApiConnection(api_key=SecretStr("kb-key")),
        api_keys={"claude": SecretStr("host-token")},
    )


def test_nested_connection_reads_the_kb_mcp_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_MCP__API_KEYS", "claude:host-token")
    monkeypatch.setenv("KB_MCP__KB_API__BASE_URL", "http://kb.internal:8000")
    monkeypatch.setenv("KB_MCP__KB_API__API_KEY", "nested-key")

    settings = KbMcpSettings()

    assert settings.kb_api.base_url == "http://kb.internal:8000"
    assert settings.kb_api.api_key.get_secret_value() == "nested-key"


def test_a_stray_kb_client_key_cannot_configure_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak `KbApiConnection`'s prefix override exists to close.

    `KB_MCP__KB_API__BASE_URL` is set and the API key is not, which is the shape
    that made this reachable: the parent's environment source produces a
    `kb_api` mapping, the nested model then runs its own source over it, and
    with the inherited prefix `KB_CLIENT__API_KEY` completed the model. It must
    now fail instead, naming the variable the operator actually has to set.
    """
    monkeypatch.setenv("KB_MCP__API_KEYS", "claude:host-token")
    monkeypatch.setenv("KB_MCP__KB_API__BASE_URL", "http://kb.internal:8000")
    monkeypatch.setenv("KB_CLIENT__API_KEY", "leaked-from-another-tool")

    with pytest.raises(ConfigurationError) as caught:
        KbMcpSettings()

    assert "KB_MCP__KB_API__API_KEY is required but not set" in str(caught.value)


def test_the_nested_variable_wins_over_kb_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_MCP__API_KEYS", "claude:host-token")
    monkeypatch.setenv("KB_MCP__KB_API__API_KEY", "nested-key")
    monkeypatch.setenv("KB_CLIENT__API_KEY", "leaked-from-another-tool")

    assert KbMcpSettings().kb_api.api_key.get_secret_value() == "nested-key"


def test_api_keys_use_the_shared_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_MCP__KB_API__API_KEY", "kb-key")
    monkeypatch.setenv("KB_MCP__API_KEYS", "claude:one, reader:two")
    monkeypatch.setenv("KB_MCP__API_KEY_SCOPES", "reader:search,claude:search|write")

    settings = KbMcpSettings()

    assert set(settings.api_keys) == {"claude", "reader"}
    assert settings.api_keys["reader"].get_secret_value() == "two"
    assert settings.api_key_scopes["claude"] == frozenset({"search", "write"})


def test_a_malformed_key_string_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_MCP__KB_API__API_KEY", "kb-key")
    monkeypatch.setenv("KB_MCP__API_KEYS", "no-separator")

    with pytest.raises(ConfigurationError) as caught:
        KbMcpSettings()

    assert "KB_MCP__API_KEYS is invalid" in str(caught.value)


def test_inbound_keys_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_MCP__KB_API__API_KEY", "kb-key")

    with pytest.raises(ConfigurationError) as caught:
        KbMcpSettings()

    assert "KB_MCP__API_KEYS is required but not set" in str(caught.value)


def test_defaults() -> None:
    settings = _minimal()

    assert settings.port == 9000
    assert settings.allow_ingest is True
    assert settings.api_key_scopes == {}
    # Unset means the RFC 9728 route is not served at all, which is right
    # locally and is what a deployment overrides.
    assert settings.resource_server_url is None


def test_the_connection_keeps_its_inherited_behaviour() -> None:
    connection = KbApiConnection(base_url="http://kb.test/", api_key="kb-key")

    assert connection.base_url == "http://kb.test"
    assert connection.user_agent.startswith("kb-mcp/")
