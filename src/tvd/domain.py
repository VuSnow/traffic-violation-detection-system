"""Plain data shared by every component. Nothing here depends on a model or a tracking library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in pixels, from its top-left corner (x1, y1) to its bottom-right corner (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Box corners are reversed: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: Box) -> float:
        """Intersection over union: 1 for the same box, 0 for boxes that do not touch."""
        width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class Detection:
    """One object found in one frame by a detector."""

    box: Box
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class TrackedObject:
    """One object in one frame, with the identity a tracker gave it across frames."""

    track_id: int
    box: Box
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True, eq=False)
class Frame:
    """One decoded video frame. ``eq=False``: comparing two images pixel by pixel is never what a test means."""

    index: int
    timestamp_s: float
    image: np.ndarray
