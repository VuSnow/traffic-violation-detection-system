"""Settings files in YAML, checked against models, so that a wrong key or value is reported with its file and its place."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(ValueError):
    """A settings file that cannot be used, with every problem found in it."""


class StrictModel(BaseModel):
    """Refuses keys it does not know, so that a misspelt setting is never silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentConfig(StrictModel):
    """Which implementation to build, by its registered name, and the options given to it."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def no_options_when_left_empty(cls, value: Any) -> Any:
        """``options:`` with nothing under it reads as null in YAML, and means no options."""
        return {} if value is None else value


class PerceptionConfig(StrictModel):
    """How to detect and track."""

    detector: ComponentConfig
    tracker: ComponentConfig
    track_classes: list[str] = Field(min_length=1)


M = TypeVar("M", bound=BaseModel)


def describe(error: ValidationError, where: str = "") -> str:
    """Every problem found by pydantic, as place: message, joined by ; . where is put before each place."""

    def place(location: tuple) -> str:
        return ".".join([where] * bool(where) + [str(part) for part in location]) or "(file)"

    return "; ".join(f"{place(e['loc'])}: {e['msg']}" for e in error.errors())


def validate(model: type[M], data: Any, where: str = "") -> M:
    """Check data already read, such as one section of a file, against ``model``. Problems raise a ``ConfigError``."""
    try:
        return model.model_validate(data if data is not None else {})
    except ValidationError as error:
        raise ConfigError(describe(error, where)) from error


def load_config(path: str | Path, model: type[M]) -> M:
    """Read a YAML file and check it against ``model``. Every problem is reported at once, in a ``ConfigError``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Settings file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: not valid YAML: {error}") from error
    try:
        return validate(model, data)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error


def file_version(path: str | Path) -> str:
    """A settings file's version, from its content: any change to the file changes it. Records keep it."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
