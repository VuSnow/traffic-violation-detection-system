"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame

ROOT = Path(__file__).resolve().parents[2]


@DETECTORS.register("test-moving-car")
class MovingCarDetector(BaseDetector):
    """A car that moves right by ``step`` px per frame, and a person who stands still."""

    def __init__(self, step: float = 5.0) -> None:
        self.step = step

    def detect(self, frame: Frame) -> list[Detection]:
        x = 10 + self.step * frame.index
        return [Detection(Box(x, 50, x + 40, 80), 2, "car", 0.9), Detection(Box(0, 0, 10, 30), 0, "person", 0.8)]


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
