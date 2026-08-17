"""`HttpServiceSettings` — what the operator types, and what happens when they
type it wrong. The environment is emptied first so a developer's own `.env`
cannot change an outcome.

The key format itself belongs to `platform_core.auth` and is tested there, in
every malformed shape. What these assert is that this class still declares those
fields — a `KB_API__API_KEYS` that stopped parsing would fail here too."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from platform_core import ConfigurationError
from platform_fastapi import HttpServiceSettings


class ExampleSettings(HttpServiceSettings):
    model_config = SettingsConfigDict(env_prefix="EXAMPLE__")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in [n for n in os.environ if n.startswith("EXAMPLE__")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


def test_api_keys_parse_from_a_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "n8n:secret-one, cli:secret-two")
    settings = ExampleSettings()

    assert set(settings.api_keys) == {"n8n", "cli"}
    assert settings.api_keys["n8n"].get_secret_value() == "secret-one"
    assert settings.api_keys["cli"].get_secret_value() == "secret-two"


def test_malformed_api_keys_name_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "no-separator")
    with pytest.raises(ConfigurationError) as caught:
        ExampleSettings()
    assert "EXAMPLE__API_KEYS" in str(caught.value)


def test_api_keys_are_required() -> None:
    with pytest.raises(ConfigurationError) as caught:
        ExampleSettings()
    assert "EXAMPLE__API_KEYS is required but not set" in str(caught.value)


def test_cors_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "n8n:secret")
    assert ExampleSettings().cors_origins == ()


def test_cors_origins_split_on_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "n8n:secret")
    monkeypatch.setenv("EXAMPLE__CORS_ORIGINS", "https://a.example, https://b.example")
    assert ExampleSettings().cors_origins == ("https://a.example", "https://b.example")


def test_health_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "n8n:secret")
    monkeypatch.setenv("EXAMPLE__HEALTH_TIMEOUT_SECONDS", "0")
    with pytest.raises(ConfigurationError) as caught:
        ExampleSettings()
    assert "EXAMPLE__HEALTH_TIMEOUT_SECONDS is invalid" in str(caught.value)


def test_constructed_in_process_from_a_mapping() -> None:
    settings = ExampleSettings(api_keys={"n8n": SecretStr("secret")})
    assert settings.api_keys["n8n"].get_secret_value() == "secret"


def test_defaults_carry_through_from_the_core_base() -> None:
    settings = ExampleSettings(api_keys={"n8n": SecretStr("secret")})
    assert (settings.env, settings.log_level, settings.log_json) == ("local", "INFO", True)
    assert settings.health_timeout_seconds == 2.0


# --- scopes (AD-024) ------------------------------------------------------


def test_scopes_parse_from_a_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "n8n:one,cli:two")
    monkeypatch.setenv("EXAMPLE__API_KEY_SCOPES", "n8n:search, cli:search|write")
    settings = ExampleSettings()

    assert settings.api_key_scopes == {
        "n8n": frozenset({"search"}),
        "cli": frozenset({"search", "write"}),
    }


def test_scopes_default_to_empty_meaning_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE__API_KEYS", "n8n:one")
    assert ExampleSettings().api_key_scopes == {}
