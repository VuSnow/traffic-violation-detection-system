"""Settings files in YAML, checked against models, so that a wrong key or value is reported with its file and its place."""

from __future__ import annotations

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
        return model.model_validate(data if data is not None else {})
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(str(part) for part in e['loc']) or '(file)'}: {e['msg']}" for e in error.errors())
        raise ConfigError(f"{path}: {problems}") from error
