"""Key resolution, scopes, and the configuration format they are typed in.

Nothing here imports a web framework: these are the primitives `kb-api` reaches
through `platform-fastapi` and `kb-mcp` reaches directly, and a test that needed
an app to exercise them would be testing the wrong layer.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict

from platform_core import (
    ApiKeyRegistry,
    ApiKeys,
    ApiKeyScopes,
    AuthenticationError,
    BaseServiceSettings,
    ConfigurationError,
    Principal,
)


class KeyedSettings(BaseServiceSettings):
    """The declaration a service makes: two annotated fields, no validators."""

    model_config = SettingsConfigDict(env_prefix="ONE__")

    api_keys: ApiKeys
    api_key_scopes: ApiKeyScopes = Field(default_factory=dict)


class OtherKeyedSettings(KeyedSettings):
    """A second service, differing only in prefix."""

    model_config = SettingsConfigDict(env_prefix="TWO__")


_MANAGED_PREFIXES = ("ONE__", "TWO__")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in [n for n in os.environ if n.startswith(_MANAGED_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


# --- the registry ---------------------------------------------------------


def test_registry_resolves_the_matching_key() -> None:
    registry = ApiKeyRegistry({"n8n": SecretStr("one"), "cli": SecretStr("two")})

    assert registry.resolve("one") == Principal(key_id="n8n")
    assert registry.resolve("two") == Principal(key_id="cli")


def test_registry_rejects_an_unknown_key() -> None:
    registry = ApiKeyRegistry({"n8n": SecretStr("one")})

    with pytest.raises(AuthenticationError) as caught:
        registry.resolve("two")
    assert caught.value.status_code == 401


def test_registry_rejects_a_prefix_of_a_real_key() -> None:
    registry = ApiKeyRegistry({"n8n": SecretStr("secret-value")})

    with pytest.raises(AuthenticationError):
        registry.resolve("secret")


def test_identify_returns_none_rather_than_raising() -> None:
    """The rate limiter asks before the auth dependency has run, and must not
    turn an unrecognised key into a failure of its own."""
    registry = ApiKeyRegistry({"n8n": SecretStr("one")})

    assert registry.identify("one") == Principal(key_id="n8n")
    assert registry.identify("nope") is None


# --- scopes (AD-024) ------------------------------------------------------


def test_registry_carries_the_grants_it_was_given() -> None:
    registry = ApiKeyRegistry(
        {"n8n": SecretStr("one"), "cli": SecretStr("two")},
        {"n8n": frozenset({"search"})},
    )

    assert registry.resolve("one") == Principal(key_id="n8n", scopes=frozenset({"search"}))
    # No entry, so unrestricted — `None`, not an empty set.
    assert registry.resolve("two") == Principal(key_id="cli", scopes=None)


def test_grants_expose_every_key_for_the_startup_log() -> None:
    """A composition root logs these, which is how the permissive default above
    is visible rather than assumed."""
    registry = ApiKeyRegistry(
        {"n8n": SecretStr("one"), "cli": SecretStr("two")},
        {"n8n": frozenset({"search"})},
    )

    assert registry.grants == (("n8n", frozenset({"search"})), ("cli", None))


@pytest.mark.parametrize(
    ("scopes", "scope", "expected"),
    [
        (None, "write", True),
        (frozenset({"search", "write"}), "write", True),
        (frozenset({"search"}), "write", False),
        (frozenset(), "search", False),
    ],
)
def test_has_scope(scopes: frozenset[str] | None, scope: str, expected: bool) -> None:
    assert Principal(key_id="k", scopes=scopes).has_scope(scope) is expected


# --- the configuration format ---------------------------------------------


def test_api_keys_parse_from_a_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONE__API_KEYS", "n8n:secret-one, cli:secret-two")
    settings = KeyedSettings()

    assert set(settings.api_keys) == {"n8n", "cli"}
    assert settings.api_keys["n8n"].get_secret_value() == "secret-one"
    assert settings.api_keys["cli"].get_secret_value() == "secret-two"


def test_secrets_do_not_appear_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONE__API_KEYS", "n8n:secret-one")
    settings = KeyedSettings()

    assert "secret-one" not in repr(settings)
    assert "secret-one" not in str(settings.api_keys["n8n"])


@pytest.mark.parametrize("value", ["", "no-separator", "n8n:", ":secret", " , "])
def test_malformed_api_keys_name_the_variable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ONE__API_KEYS", value)
    with pytest.raises(ConfigurationError) as caught:
        KeyedSettings()
    assert "ONE__API_KEYS" in str(caught.value)


def test_a_secret_containing_a_colon_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why scopes are their own setting rather than a third field on `api_keys`:
    the key parser splits once, so a colon in the secret is not ambiguous — but
    it would be if scopes came after it."""
    monkeypatch.setenv("ONE__API_KEYS", "cli:aa:bb:cc")
    assert KeyedSettings().api_keys["cli"].get_secret_value() == "aa:bb:cc"


def test_a_mapping_passes_through_unparsed() -> None:
    """In-process wiring and tests construct these from a mapping; only the
    environment arrives as a string."""
    settings = KeyedSettings(
        api_keys={"n8n": SecretStr("secret")}, api_key_scopes={"n8n": frozenset({"search"})}
    )

    assert settings.api_keys["n8n"].get_secret_value() == "secret"
    assert settings.api_key_scopes == {"n8n": frozenset({"search"})}


def test_scopes_parse_from_a_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONE__API_KEYS", "n8n:one,cli:two")
    monkeypatch.setenv("ONE__API_KEY_SCOPES", "n8n:search, cli:search|write")

    assert KeyedSettings().api_key_scopes == {
        "n8n": frozenset({"search"}),
        "cli": frozenset({"search", "write"}),
    }


@pytest.mark.parametrize("value", ["n8n", "n8n:", ":search"])
def test_malformed_scopes_are_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ONE__API_KEYS", "n8n:one")
    monkeypatch.setenv("ONE__API_KEY_SCOPES", value)

    with pytest.raises(ConfigurationError):
        KeyedSettings()


def test_two_services_read_the_same_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the format lives here rather than on one service's settings:
    `KB_MCP__API_KEYS` has to mean what `KB_API__API_KEYS` means, down to the
    colon in a secret and the rejection of an empty scope list."""
    for prefix in ("ONE", "TWO"):
        monkeypatch.setenv(f"{prefix}__API_KEYS", "cli:aa:bb")
        monkeypatch.setenv(f"{prefix}__API_KEY_SCOPES", "cli:search|write")

    one, two = KeyedSettings(), OtherKeyedSettings()

    assert (
        one.api_keys["cli"].get_secret_value() == two.api_keys["cli"].get_secret_value() == "aa:bb"
    )
    assert one.api_key_scopes == two.api_key_scopes == {"cli": frozenset({"search", "write"})}

    monkeypatch.setenv("TWO__API_KEY_SCOPES", "cli:")
    with pytest.raises(ConfigurationError) as caught:
        OtherKeyedSettings()
    assert "TWO__API_KEY_SCOPES" in str(caught.value)
