"""How each candidate's evidence is cut and kept: ``configs/tvd/evidence.yaml``. There are no defaults in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tvd.config import StrictModel, load_config


class ClipSettings(StrictModel):
    pre_seconds: float = Field(ge=0)
    max_pre_seconds: float = Field(ge=0)
    post_seconds: float = Field(ge=0)
    crf: int = Field(ge=0, le=51)


class ImageSettings(StrictModel):
    offset_seconds: float = Field(gt=0)
    jpeg_quality: int = Field(ge=1, le=100)
    box_tolerance_frames: int = Field(ge=0)


class RecordSettings(StrictModel):
    duplicate_seconds: float = Field(ge=0)


class EvidenceSettings(StrictModel):
    version: int
    clip: ClipSettings
    images: ImageSettings
    records: RecordSettings


def load_evidence_settings(path: str | Path) -> EvidenceSettings:
    return load_config(path, EvidenceSettings)
