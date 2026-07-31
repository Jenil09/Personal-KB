"""Settings loader behaviour.

Every test runs in an empty temporary directory with the relevant environment
variables cleared, so neither a developer's real `.env` nor their shell can
change the outcome.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import SettingsConfigDict

from platform_core.errors import ConfigurationError
from platform_core.settings import BaseServiceSettings


class ProviderSettings(BaseModel):
    api_key: str
    timeout_seconds: float = 30.0


class ServiceSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_TEST__")

    api_keys: str
    openai: ProviderSettings


class UnprefixedSettings(BaseServiceSettings):
    api_keys: str


class FlatSettings(BaseServiceSettings):
    """No nesting delimiter, to prove the env-var reconstruction still works."""

    model_config = SettingsConfigDict(env_prefix="KB_FLAT__", env_nested_delimiter=None)

    openai: ProviderSettings


class LimitSettings(BaseModel):
    """A group with nothing required — there is no leaf to point the operator at."""

    per_minute: int = 60


class CollectionSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_TEST__")

    providers: list[ProviderSettings]
    limits: LimitSettings


_MANAGED_PREFIXES = ("KB_TEST__", "KB_FLAT__", "API_KEYS", "ENV", "LOG_")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """A known-empty environment and no `.env` on disk.

    The unprefixed class reads bare names like `API_KEYS`, and `env_file` is
    resolved relative to the working directory, so both need neutralising.
    """
    for name in [n for n in os.environ if n.startswith(_MANAGED_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


def _service(monkeypatch: pytest.MonkeyPatch) -> ServiceSettings:
    monkeypatch.setenv("KB_TEST__API_KEYS", "local-dev:secret")
    monkeypatch.setenv("KB_TEST__OPENAI__API_KEY", "sk-test")
    return ServiceSettings()


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _service(monkeypatch)
    assert settings.env == "local"
    assert settings.log_level == "INFO"
    assert settings.log_json is True


def test_prefix_and_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_TEST__ENV", "production")
    monkeypatch.setenv("KB_TEST__LOG_LEVEL", "WARNING")
    monkeypatch.setenv("KB_TEST__OPENAI__TIMEOUT_SECONDS", "5")
    settings = _service(monkeypatch)

    assert settings.env == "production"
    assert settings.log_level == "WARNING"
    assert settings.openai.api_key == "sk-test"
    assert settings.openai.timeout_seconds == 5.0


def test_env_file_is_loaded_from_the_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text("KB_TEST__API_KEYS=from-file\nKB_TEST__OPENAI__API_KEY=sk-file\n")
    settings = ServiceSettings()
    assert settings.api_keys == "from-file"
    assert settings.openai.api_key == "sk-file"

    monkeypatch.setenv("KB_TEST__API_KEYS", "from-env")
    assert ServiceSettings().api_keys == "from-env", "the environment must win over .env"


def test_unprefixed_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefixed class must not pick up a bare variable of the same name."""
    monkeypatch.setenv("API_KEYS", "leaked")
    settings = _service(monkeypatch)
    assert settings.api_keys == "local-dev:secret"


def test_missing_secret_names_the_environment_variable() -> None:
    with pytest.raises(ConfigurationError) as caught:
        ServiceSettings()

    message = caught.value.detail
    assert "ServiceSettings is misconfigured" in message
    assert "KB_TEST__API_KEYS is required but not set" in message
    assert "KB_TEST__OPENAI__API_KEY is required but not set" in message
    assert caught.value.context["field_errors"] == [
        "KB_TEST__API_KEYS",
        "KB_TEST__OPENAI__API_KEY",
    ]


def test_missing_secret_without_prefix() -> None:
    with pytest.raises(ConfigurationError) as caught:
        UnprefixedSettings()
    assert "API_KEYS is required but not set" in caught.value.detail


def test_env_var_name_without_nesting_delimiter() -> None:
    with pytest.raises(ConfigurationError) as caught:
        FlatSettings()
    assert "KB_FLAT__OPENAI_API_KEY is required but not set" in caught.value.detail


def test_missing_group_with_no_required_fields_falls_back_to_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_TEST__PROVIDERS", '[{"api_key": "sk-test"}]')
    with pytest.raises(ConfigurationError) as caught:
        CollectionSettings()
    assert "KB_TEST__LIMITS is required but not set" in caught.value.detail


def test_missing_field_inside_a_list_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """An indexed location cannot be walked through the model, so it is reported as-is."""
    monkeypatch.setenv("KB_TEST__PROVIDERS", '[{"timeout_seconds": 1}]')
    monkeypatch.setenv("KB_TEST__LIMITS", "{}")
    with pytest.raises(ConfigurationError) as caught:
        CollectionSettings()
    assert "KB_TEST__PROVIDERS__0__API_KEY is required but not set" in caught.value.detail


def test_invalid_value_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_TEST__LOG_LEVEL", "chatty")
    with pytest.raises(ConfigurationError) as caught:
        _service(monkeypatch)
    assert "KB_TEST__LOG_LEVEL is invalid:" in caught.value.detail


def test_configuration_error_chains_the_pydantic_cause() -> None:
    with pytest.raises(ConfigurationError) as caught:
        ServiceSettings()
    assert isinstance(caught.value.__cause__, PydanticValidationError)


def test_settings_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _service(monkeypatch)
    with pytest.raises(PydanticValidationError):
        # mypy knows this is read-only; the assertion is that it also fails at
        # runtime, where a service actually would.
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_unknown_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_TEST__NOT_A_FIELD", "whatever")
    assert _service(monkeypatch).api_keys == "local-dev:secret"
