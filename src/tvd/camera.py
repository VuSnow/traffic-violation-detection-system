"""Camera files: what one camera sees, and which rules apply to it. One file per camera, never in code.

The defaults of every rule are in one file for all cameras. A camera file may change any of them for
that camera only, under its rule's ``settings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, RootModel, field_validator

from tvd.config import ComponentConfig, StrictModel, load_config


class RuleSection(StrictModel):
    """One rule of one camera: where things are in its image (``scene``), and its own settings, if any."""

    scene: dict[str, Any]
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("settings", mode="before")
    @classmethod
    def no_settings_when_left_empty(cls, value: Any) -> Any:
        return {} if value is None else value


class CameraConfig(StrictModel):
    camera_id: str
    frame_size: tuple[int, int]  # Width and height. A video of another size is refused.
    motion: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="fixed"))
    rules: dict[str, RuleSection] = Field(min_length=1)


class RuleDefaults(RootModel[dict[str, dict[str, Any]]]):
    """The settings of each rule, by rule name."""


def load_camera(path: str | Path) -> CameraConfig:
    return load_config(path, CameraConfig)


def load_rule_defaults(path: str | Path) -> dict[str, dict[str, Any]]:
    return load_config(path, RuleDefaults).root


def effective_settings(defaults: dict[str, dict[str, Any]], section: RuleSection, rule: str) -> dict[str, Any]:
    """A rule's defaults, with what the camera file changes for this camera. The rule checks the result."""
    return {**defaults.get(rule, {}), **section.settings}
