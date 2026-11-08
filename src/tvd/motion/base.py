"""Camera motion: how each frame's pixels map to the pixels of one reference frame.

A camera file draws its stop line and regions on the reference frame. When the camera moves, each
frame needs its own transform to compare a vehicle's position with them. The transform of a frame is a
similarity ``(a, b, tx, ty)``: ``x' = a*x - b*y + tx``, ``y' = b*x + a*y + ty``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import cv2
import numpy as np

from tvd.domain import Detection
from tvd.registry import Registry
from tvd.video import VideoReader

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def similarity_matrix(params: Sequence[float]) -> np.ndarray:
    """The 2x3 matrix of a similarity transform given as ``(a, b, tx, ty)``."""
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


class CameraMotion:
    """The transform of every analysed frame, from frame pixels to reference pixels.

    Built without ``params``, it is a fixed camera: points are returned unchanged, with no arithmetic,
    so that a fixed camera's results never depend on rounding.
    """

    def __init__(self, params: np.ndarray | None = None, first_frame: int = 0) -> None:
        self._params = None if params is None else np.asarray(params, dtype=np.float64).reshape(-1, 4)
        self.first_frame = first_frame

    @property
    def is_fixed(self) -> bool:
        return self._params is None

    def params_at(self, frame_index: int) -> np.ndarray:
        if self._params is None:
            return np.array(IDENTITY)
        offset = frame_index - self.first_frame
        if not 0 <= offset < len(self._params):
            raise IndexError(f"Frame {frame_index} is outside the {len(self._params)} frames from {self.first_frame}")
        return self._params[offset]

    def to_reference(self, frame_index: int, points) -> np.ndarray:
        """``(x, y)`` points of a frame, in reference pixels."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._params is None:
            return points
        return cv2.transform(points[None], similarity_matrix(self.params_at(frame_index)))[0]

    def to_frame(self, frame_index: int, points) -> np.ndarray:
        """Reference points, in the pixels of a frame: where to draw a stop line on that frame."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._params is None:
            return points
        inverse = cv2.invertAffineTransform(similarity_matrix(self.params_at(frame_index)))
        return cv2.transform(points[None], inverse)[0]


class BaseMotionEstimator(ABC):
    """Finds how the camera moved in each frame of a range."""

    needs_detections: bool = False
    """True if ``estimate`` needs every frame's detections, to leave moving objects out."""

    @abstractmethod
    def estimate(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> CameraMotion:
        """The motion of frames ``start`` to ``stop`` (not included)."""

    def check_frames(self, frame_count: int) -> None:
        """Raise ``ValueError`` if a video of ``frame_count`` frames cannot be used, before any work. Any length can, by default."""


MOTION_ESTIMATORS: Registry[BaseMotionEstimator] = Registry("camera motion estimator")
