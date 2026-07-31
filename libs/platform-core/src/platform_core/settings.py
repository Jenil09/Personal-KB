"""Settings base class shared by every service.

Each service subclasses `BaseServiceSettings` and sets its own `env_prefix`
(`KB_API__`); the rest of the config — `__` nesting, `.env` loading,
immutability — is inherited.

Secrets carry no defaults. A missing one therefore fails when settings are
constructed at startup, and the message names the environment variable rather
than the pydantic field path, because the variable is what the operator has to
go and set.
"""

from typing import Any, Literal

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from platform_core.errors import ConfigurationError

__all__ = ["BaseServiceSettings", "Environment", "LogLevel"]

Environment = Literal["local", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class BaseServiceSettings(BaseSettings):
    """Cross-service configuration. Subclasses add their own fields.

    Immutable once constructed: settings are read at startup and passed down
    explicitly, so nothing should be mutating them later.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    env: Environment = "local"
    log_level: LogLevel = "INFO"
    log_json: bool = True

    def __init__(self, **values: Any) -> None:
        # `Any` to stay compatible with BaseSettings' own kwargs (`_env_file`
        # and friends) without restating them.
        try:
            super().__init__(**values)
        except PydanticValidationError as exc:
            problems = _problems(type(self), exc)
            lines = [f"{type(self).__name__} is misconfigured:"]
            lines += [f"  {var} {complaint}" for var, complaint in problems]
            raise ConfigurationError(
                "\n".join(lines),
                context={"field_errors": [var for var, _ in problems]},
            ) from exc


def _problems(cls: type[BaseSettings], exc: PydanticValidationError) -> list[tuple[str, str]]:
    """Turn pydantic's field errors into (environment variable, complaint) pairs."""
    problems: list[tuple[str, str]] = []
    for error in exc.errors():
        loc = tuple(str(part) for part in error["loc"])
        if error["type"] == "missing":
            # An absent nested group is reported by pydantic as one error on
            # the group itself; expand it, because the operator needs the leaf
            # variable names, not the name of the model.
            problems += [
                (_env_var(cls, leaf), "is required but not set")
                for leaf in _missing_leaves(_annotation_at(cls, loc), loc)
            ]
        else:
            problems.append((_env_var(cls, loc), f"is invalid: {error['msg']}"))
    return problems


def _env_var(cls: type[BaseSettings], loc: tuple[str, ...]) -> str:
    """Reconstruct the environment variable a field is populated from."""
    prefix = cls.model_config.get("env_prefix") or ""
    delimiter = cls.model_config.get("env_nested_delimiter") or "_"
    return f"{prefix}{delimiter.join(loc)}".upper()


def _annotation_at(cls: type[BaseSettings], loc: tuple[str, ...]) -> object:
    """The annotation `loc` points at, or `None` if it cannot be resolved."""
    annotation: object = cls
    for part in loc:
        fields = getattr(annotation, "model_fields", None)
        if fields is None or part not in fields:
            return None
        annotation = fields[part].annotation
    return annotation


def _missing_leaves(annotation: object, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every required leaf path under `annotation`, or `prefix` if it is a scalar."""
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        return [prefix]
    leaves = [
        leaf
        for name, field in annotation.model_fields.items()
        if field.is_required()
        for leaf in _missing_leaves(field.annotation, (*prefix, name))
    ]
    return leaves or [prefix]
