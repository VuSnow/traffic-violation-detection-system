"""How plates are read: ``configs/tvd/plates.yaml``. There are no defaults in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tvd.config import ComponentConfig, StrictModel, load_config


class ReadingSettings(StrictModel):
    frames_per_track: int = Field(ge=1)
    min_frame_gap: int = Field(ge=0)
    vehicle_margin: float = Field(ge=0)
    lowest_plate_center: float = Field(ge=0, le=1)
    min_plate_width: int = Field(ge=1)
    min_chars: int = Field(ge=1)
    min_char_prob: float = Field(ge=0, le=1)
    min_votes: int = Field(ge=1)
    min_agreement: float = Field(ge=0, le=1)


class PlatesConfig(StrictModel):
    version: int
    models: ComponentConfig
    reading: ReadingSettings


def load_plates_config(path: str | Path) -> PlatesConfig:
    return load_config(path, PlatesConfig)
