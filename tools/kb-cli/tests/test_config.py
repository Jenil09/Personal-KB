"""Configuration precedence, storage, and the file's permissions.

The behaviour that matters is the ordering — environment over stored file over
default — because it is what an operator relies on to point a configured CLI at
a local stack for one command. The rest is making sure a secret written to disk
is written the way a secret should be.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from kb_cli.config import (
    CONFIG_PATH_ENV_VAR,
    KbCliSettings,
    config_path,
    load_config_file,
    load_settings,
    redact,
    save_config_file,
)
from platform_core import ConfigurationError


def test_environment_overrides_the_stored_file(monkeypatch: pytest.MonkeyPatch) -> None:
    save_config_file({"base_url": "http://stored:8000", "api_key": "stored-key"})
    monkeypatch.setenv("KB_CLI__BASE_URL", "http://from-env:9000")

    settings = load_settings()

    assert settings.base_url == "http://from-env:9000"
    # The key was not overridden, so it still comes from the file — the two
    # sources merge per field rather than one replacing the other wholesale.
    assert settings.api_key.get_secret_value() == "stored-key"


def test_stored_file_overrides_the_default() -> None:
    save_config_file({"base_url": "http://stored:8000", "api_key": "k"})

    assert load_settings().base_url == "http://stored:8000"


def test_defaults_apply_when_nothing_is_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_CLI__API_KEY", "from-env")

    settings = load_settings()

    assert settings.base_url == "http://localhost:8000"
    assert settings.timeout_seconds == 30.0


def test_a_missing_api_key_fails_at_construction() -> None:
    """No default for a secret, so this is a startup failure and not a 401."""
    with pytest.raises(ConfigurationError) as caught:
        load_settings()

    assert "KB_CLI__API_KEY" in str(caught.value)


def test_settings_are_read_fresh_each_time() -> None:
    """`load_settings` must not cache: the settings screen writes mid-process."""
    save_config_file({"api_key": "first", "base_url": "http://one:8000"})
    assert load_settings().base_url == "http://one:8000"

    save_config_file({"api_key": "first", "base_url": "http://two:8000"})
    assert load_settings().base_url == "http://two:8000"


def test_config_path_honours_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.json"
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(target))

    assert config_path() == target


def test_config_path_defaults_under_the_platform_config_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)

    resolved = config_path()

    assert resolved.name == "config.json"
    assert resolved.parent.name == "kb-cli"


def test_saving_creates_the_directory_and_round_trips() -> None:
    values = {"base_url": "http://kb.test", "api_key": "secret", "timeout_seconds": 12.5}

    written = save_config_file(values)

    assert written.exists()
    assert load_config_file() == values


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_stored_file_is_not_readable_by_others() -> None:
    """It holds an API key in cleartext; `0600` is what keeps that private."""
    written = save_config_file({"api_key": "secret"})

    mode = stat.S_IMODE(written.stat().st_mode)

    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_saving_leaves_no_temporary_file_behind() -> None:
    written = save_config_file({"api_key": "secret"})

    assert list(written.parent.iterdir()) == [written]


def test_a_missing_file_reads_as_empty() -> None:
    assert load_config_file() == {}


def test_malformed_json_is_an_error_rather_than_a_silent_default() -> None:
    """Falling back would point the CLI at the wrong service without saying so."""
    target = config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config_file()


def test_a_json_scalar_is_rejected() -> None:
    target = config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps("just a string"), encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config_file()


def test_a_trailing_slash_on_the_base_url_is_normalised() -> None:
    """Otherwise `httpx` joins the `/v1` prefix differently for the two forms."""
    assert KbCliSettings(base_url="http://kb.test/", api_key="k").base_url == "http://kb.test"


def test_redact_masks_the_key_without_hiding_that_one_is_set() -> None:
    masked = redact({"api_key": "secret", "base_url": "http://kb.test"})

    assert masked["api_key"] == "********"
    assert masked["base_url"] == "http://kb.test"


def test_redact_leaves_an_absent_key_visibly_absent() -> None:
    assert redact({"api_key": ""})["api_key"] == ""
