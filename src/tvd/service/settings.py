"""The service's own settings: ``configs/service.yaml``. There are no defaults in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from tvd.config import StrictModel, load_config


class StorageSettings(StrictModel):
    folder: Path
    keep_days: float = Field(gt=0)
    max_upload_mb: int = Field(gt=0)


class JobSettings(StrictModel):
    deadline_minutes: float = Field(gt=0)
    seconds_per_video_second: dict[str, float] = Field(min_length=1)

    @field_validator("seconds_per_video_second")
    @classmethod
    def positive(cls, value: dict[str, float]) -> dict[str, float]:
        if any(v <= 0 for v in value.values()):
            raise ValueError("every value must be greater than 0")
        return value

    def work_seconds(self, motion_type: str, video_seconds: float) -> float:
        """How long a video takes, from the measured speed of its kind of camera."""
        return video_seconds * self.seconds_per_video_second[motion_type]

    def longest_video_seconds(self, motion_type: str) -> float:
        """The longest video of a kind of camera that can be done within the deadline, with an empty queue."""
        return self.deadline_minutes * 60 / self.seconds_per_video_second[motion_type]


class ServerSettings(StrictModel):
    host: str
    port: int = Field(ge=1, le=65535)


class SettingsFiles(StrictModel):
    cameras: Path
    perception: Path
    rules: Path
    plates: Path
    evidence: Path


class ServiceSettings(StrictModel):
    version: int
    storage: StorageSettings
    jobs: JobSettings
    server: ServerSettings
    files: SettingsFiles


def load_service_settings(path: str | Path) -> ServiceSettings:
    return load_config(path, ServiceSettings)
