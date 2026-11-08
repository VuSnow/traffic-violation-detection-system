"""Every camera the service knows: the files of one folder, all checked when the service starts.

A camera file with a mistake would only fail when a video of that camera arrives, perhaps at night.
So the service refuses to start instead, and names the file and the setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tvd.camera import CameraConfig, load_camera
from tvd.config import ConfigError, file_version
from tvd.factory import build_motion_estimator, build_rules


@dataclass(frozen=True)
class Camera:
    config: CameraConfig
    path: Path
    version: str

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def motion_type(self) -> str:
        return self.config.motion.type

    def describe(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id, "frame_size": list(self.config.frame_size), "motion": self.motion_type,
            "moving": self.motion_type != "fixed", "rules": sorted(self.config.rules),
        }


class CameraCatalog:
    """The cameras of ``folder``, one ``<camera id>.yaml`` each. ``speeds`` are the measured speeds by motion type."""

    def __init__(self, folder: Path, rule_defaults: dict[str, dict[str, Any]], speeds: dict[str, float]) -> None:
        if not folder.is_dir():
            raise ConfigError(f"Camera folder not found: {folder}")
        cameras = {}
        for path in sorted(folder.glob("*.yaml")):
            config = load_camera(path)
            if config.camera_id != path.stem:
                raise ConfigError(f"{path}: camera_id {config.camera_id!r} must be the file's name, {path.stem!r}")
            try:
                build_rules(config, rule_defaults)
                build_motion_estimator(config.motion)
            except ConfigError as error:
                raise ConfigError(f"{path}: {error}") from error
            if config.motion.type not in speeds:
                raise ConfigError(f"{path}: the service has no measured speed for a {config.motion.type!r} camera (jobs.seconds_per_video_second)")
            cameras[config.camera_id] = Camera(config, path, file_version(path))
        if not cameras:
            raise ConfigError(f"No camera file in {folder}")
        self._cameras = cameras

    def get(self, camera_id: str) -> Camera | None:
        return self._cameras.get(camera_id)

    @property
    def ids(self) -> list[str]:
        return list(self._cameras)

    def describe(self) -> list[dict[str, Any]]:
        return [camera.describe() for camera in self._cameras.values()]
