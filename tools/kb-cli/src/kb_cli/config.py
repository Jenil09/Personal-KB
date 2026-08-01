"""Where the service is and which key reaches it (AD-025).

Three sources, highest first: **environment**, then a **JSON file in the user's
platform config directory**, then defaults. The file is what makes `kb` behave
like an installed tool rather than a repo script — you configure it once with
`kb config set` or the settings screen, and it works from any directory on any
machine. The environment still wins, so CI and one-off overrides (`KB_CLI__BASE_URL=…
kb status`) need no file at all and never fight one.

`KB_CLI__` is a separate prefix from `KB_API__` on purpose. The two are read on
different machines: `KB_API__*` configures the service inside the container, and
a laptop holding a stale copy of the service's environment would otherwise point
the CLI at a Postgres DSN it cannot reach. Nothing here overlaps with a service
variable, so both can live in one shell without either shadowing the other.

`api_key` carries no default. A fresh install therefore fails when settings are
constructed rather than at the first request — the rule `BaseServiceSettings`
exists to enforce — and `__main__` turns that failure into the sentence that
says which command fixes it.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any

from platformdirs import user_config_path
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from platform_core import BaseServiceSettings, ConfigurationError

__all__ = [
    "CONFIG_PATH_ENV_VAR",
    "KbCliSettings",
    "config_path",
    "load_config_file",
    "save_config_file",
]

APP_NAME = "kb-cli"

CONFIG_PATH_ENV_VAR = "KB_CLI_CONFIG"
"""Points the whole config file somewhere else.

Single underscore, unlike every settings variable, and that is the distinction:
`KB_CLI__*` are fields *within* the configuration, this names the file the
configuration is read from. It has to be resolvable before the settings model
exists, so it cannot be one of the model's own fields.
"""

_SECRET_FIELDS = frozenset({"api_key"})


def config_path() -> Path:
    """The config file for this user, per the platform's own convention.

    `platformdirs` rather than a hand-rolled `~/.kb-cli`, so this lands where
    each OS expects it and where its backup and sync tooling already looks:

        Linux    $XDG_CONFIG_HOME/kb-cli/config.json  (~/.config/kb-cli/…)
        macOS    ~/Library/Application Support/kb-cli/config.json
        Windows  %LOCALAPPDATA%\\kb-cli\\config.json

    `roaming=False` on Windows keeps a machine-local API key out of a profile
    that syncs to a domain server.
    """
    override = os.environ.get(CONFIG_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return user_config_path(APP_NAME, appauthor=False, roaming=False) / "config.json"


def load_config_file(path: Path | None = None) -> dict[str, Any]:
    """The stored configuration, or `{}` when there is not one yet.

    A missing file is the state a fresh install is in, so it is not an error.
    A malformed one is: silently falling back to defaults would leave the
    operator looking at the wrong service wondering where their documents went.
    """
    # `Any` because this is arbitrary decoded JSON on its way into a settings
    # source that validates it.
    target = path or config_path()
    if not target.exists():
        return {}
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"{target} could not be read as JSON: {exc}",
            context={"config_path": str(target)},
        ) from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"{target} must hold a JSON object, not {type(loaded).__name__}.",
            context={"config_path": str(target)},
        )
    return loaded


def save_config_file(values: dict[str, Any], path: Path | None = None) -> Path:
    """Write the configuration, owner-readable only, and return where it went.

    The file holds an API key in cleartext, which is the same posture as
    `~/.aws/credentials` or a `.npmrc` token and the reason for the `0600`: it
    is what stops another account on a shared machine from reading it. On
    Windows `chmod` cannot express that, so the mode call is skipped rather than
    silently doing nothing meaningful — NTFS inherits the user profile's ACL,
    which already excludes other users.

    Written to a temporary file in the same directory and then renamed, so an
    interrupted write cannot leave a half-written config that fails to parse on
    every subsequent run.
    """
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if os.name != "nt":
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    temporary.replace(target)
    return target


def redact(values: dict[str, Any]) -> dict[str, Any]:
    """The configuration as it is safe to print — secrets replaced, not omitted.

    Omitting them would make a configured key and an absent one look identical
    in `kb config show`, which is the one question that view exists to answer.
    """
    return {
        key: ("********" if key in _SECRET_FIELDS and value else value)
        for key, value in values.items()
    }


class _ConfigFileSource(JsonConfigSettingsSource):
    """The JSON file, resolved at call time rather than at class definition.

    `json_file` in `model_config` is read when the class body executes, which
    would freeze the path at import and make `KB_CLI_CONFIG` — and every test
    that points at a temporary directory — a no-op.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls, json_file=config_path())


class KbCliSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="KB_CLI__", env_nested_delimiter="__")

    base_url: str = "http://localhost:8000"
    """The service root, without `/v1`.

    Locally this is the port `just run` binds. Against production it is the
    tailnet address `tailscale serve` publishes — AD-023 leaves no other way in,
    which is why this defaults to the development value rather than a hostname
    that only resolves from a tailnet-joined machine.
    """

    api_key: SecretStr

    provider: str | None = None
    """Which embedding provider's collection to work against (AD-006).

    `None` sends no `provider` field at all and lets the service apply its own
    default, which is one fewer place for the two to disagree.
    """

    timeout_seconds: float = Field(default=30.0, gt=0)

    ingest_timeout_seconds: float = Field(default=300.0, gt=0)
    """Ingest gets its own budget because it is not the same shape of request.

    A search is one embedding call; ingesting a large document is a chunked
    embed of the whole thing, and the corpus averages ~0.5 MB a file. One
    timeout covering both would be either too tight for ingest or useless for
    search.
    """

    editor: str | None = None
    """Overrides `$VISUAL`/`$EDITOR` for the compose-a-document action."""

    @field_validator("base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        # `httpx.AsyncClient(base_url=...)` joins paths differently depending on
        # whether the base ends in a slash. Normalising here means a config that
        # says `http://kb-api:8000/` and one that does not build the same URL.
        return value.rstrip("/")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment over stored file, in that order.

        The stored file is last of the real sources so that a `KB_CLI__BASE_URL`
        in the shell overrides it for one command without editing anything —
        which is how you point a configured CLI at a local stack for five
        minutes and get your production configuration back by opening a new
        terminal.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _ConfigFileSource(settings_cls),
            file_secret_settings,
        )


def load_settings() -> KbCliSettings:
    """Read configuration fresh.

    Deliberately not `lru_cache`d, unlike the service's `get_settings`. A
    service reads its environment once at startup and lives with it; this
    process writes its own configuration mid-run from the settings screen and
    has to see the result without being restarted.
    """
    return KbCliSettings()
