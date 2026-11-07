"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame, TrackedObject
from tvd.signals import SIGNAL_READERS, BaseSignalReader
from tvd.tracking import TRACKERS, BaseTracker

ROOT = Path(__file__).resolve().parents[2]


@DETECTORS.register("test-moving-car")
class MovingCarDetector(BaseDetector):
    """A car that moves right by ``step`` px per frame, and a person who stands still."""

    def __init__(self, step: float = 5.0) -> None:
        self.step = step

    def detect(self, frame: Frame) -> list[Detection]:
        x = 10 + self.step * frame.index
        return [Detection(Box(x, 50, x + 40, 80), 2, "car", 0.9), Detection(Box(0, 0, 10, 30), 0, "person", 0.8)]


@TRACKERS.register("test-same-id")
class SameIdTracker(BaseTracker):
    """Gives every detection track id 1: enough to test what surrounds a tracker."""

    def __init__(self) -> None:
        self.resets = 0

    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        return [TrackedObject(1, d.box, d.class_id, d.class_name, d.confidence) for d in detections]

    def reset(self) -> None:
        self.resets += 1


@SIGNAL_READERS.register("test-fixed-state")
class FixedStateReader(BaseSignalReader):
    """A signal head that always shows ``state``."""

    def __init__(self, state: str = "red") -> None:
        self.state = state

    def read(self, image) -> str:
        return self.state


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
