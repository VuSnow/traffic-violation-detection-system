"""One pass over a video for one camera: detection and tracking, every rule's per-frame work, the camera's motion, then the rules."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tvd.camera import CameraConfig
from tvd.config import ConfigError
from tvd.detection import BaseDetector
from tvd.motion import BaseMotionEstimator
from tvd.observers import DetectionLog, FrameObserver
from tvd.perception import track_video
from tvd.rules import BaseRule, Observation, RuleResult
from tvd.tracking import BaseTracker
from tvd.video import VideoReader


@dataclass
class CheckResult:
    video: str
    camera_id: str
    frames: int
    fps: float
    tracks: int
    seconds: float
    results: dict[str, RuleResult]
    first_frame: int = 0  # The analysed frames are first_frame to first_frame + frames.

    def as_json(self) -> dict[str, Any]:
        return {
            "video": self.video, "camera_id": self.camera_id, "first_frame": self.first_frame, "frames": self.frames, "fps": self.fps, "tracks": self.tracks,
            "seconds": self.seconds,
            "rules": {
                name: {"settings": r.settings, "verdicts": r.verdict_counts(), "findings": [f.as_json() for f in r.findings], "records": r.records}
                for name, r in self.results.items()
            },
        }


def check_frame_size(reader: VideoReader, camera: CameraConfig) -> None:
    """A camera file's geometry is drawn for one frame size: a video of another size is refused before any work."""
    if (reader.width, reader.height) != tuple(camera.frame_size):
        width, height = camera.frame_size
        raise ConfigError(f"Camera {camera.camera_id} has {width}x{height} frames, but {reader.path.name} is {reader.width}x{reader.height}")


def check_video(
    reader: VideoReader,
    camera: CameraConfig,
    detector: BaseDetector,
    tracker: BaseTracker,
    track_classes: set[str],
    motion_estimator: BaseMotionEstimator,
    rules: list[BaseRule],
    start: int = 0,
    stop: int | None = None,
    observers: Sequence[FrameObserver] = (),
) -> CheckResult:
    """Detect and track once, and give every rule the same tracks. Build new rules for each video: their observers keep what they saw.

    ``observers`` see every frame too, as the rules' own do: a job's progress, for example.
    """
    check_frame_size(reader, camera)
    began = time.perf_counter()
    log = DetectionLog() if motion_estimator.needs_detections else None
    everyone = [o for rule in rules for o in rule.frame_observers()] + ([log] if log else []) + list(observers)
    history = track_video(reader, detector, tracker, track_classes, start, stop, everyone)
    end = start + history.frames_seen
    motion = motion_estimator.estimate(reader, log.by_frame if log else {}, start, end)
    observation = Observation(history.tracks, motion, reader.fps, reader.width, reader.height, start, end)
    results = {rule.name: rule.evaluate(observation) for rule in rules}
    seconds = round(time.perf_counter() - began, 1)
    return CheckResult(reader.path.name, camera.camera_id, history.frames_seen, reader.fps, len(history.tracks), seconds, results, start)
