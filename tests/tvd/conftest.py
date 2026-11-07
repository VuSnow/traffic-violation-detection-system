"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame, TrackedObject
from tvd.plates import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.signals import SIGNAL_READERS, BaseSignalReader
from tvd.tracking import TRACKERS, BaseTracker

ROOT = Path(__file__).resolve().parents[2]


@DETECTORS.register("test-moving-car")
class MovingCarDetector(BaseDetector):
    """A car that moves by ``step`` px to the right and ``step_y`` px down per frame, and a person who stands still."""

    def __init__(self, step: float = 5.0, step_y: float = 0.0) -> None:
        self.step = step
        self.step_y = step_y

    def detect(self, frame: Frame) -> list[Detection]:
        x, y = 10 + self.step * frame.index, 50 + self.step_y * frame.index
        return [Detection(Box(x, y, x + 40, y + 30), 2, "car", 0.9), Detection(Box(0, 0, 10, 30), 0, "person", 0.8)]


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


@PLATE_MODELS.register("test-plate")
class FakePlateModels(BasePlateModels):
    """Finds one plate in the lower middle of every image, a quarter of its width, and reads ``text`` on it with confidence ``prob``."""

    def __init__(self, text: str = "TEST1234", prob: float = 0.99) -> None:
        self.text = text
        self.prob = prob
        self.closed = False

    def detect(self, image) -> list[PlateBox]:
        height, width = image.shape[:2]
        return [PlateBox(Box(0.375 * width, 0.8 * height, 0.625 * width, 0.9 * height), 0.9)]

    def recognize(self, plate) -> PlateText:
        characters = self.text.ljust(9, PAD)
        return PlateText(characters, tuple(self.prob if c != PAD else 1.0 for c in characters))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
