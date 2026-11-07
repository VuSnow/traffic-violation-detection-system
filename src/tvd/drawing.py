"""Drawing for a reviewer, on evidence images only: a clip keeps its original pixels."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from tvd.domain import Box

VEHICLE_COLOR = (0, 255, 255)  # BGR: yellow


class FrameOverlay(ABC):
    """What a rule saw at one frame of the source video, drawn on an image of that frame, such as the stop line and the signal state."""

    @abstractmethod
    def draw(self, image: np.ndarray, frame_index: int) -> None:
        """Draw on ``image`` in place. ``frame_index`` is the frame of the source video, not of a clip."""


def draw_vehicle_box(image: np.ndarray, box: Box) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box.as_list())
    cv2.rectangle(image, (x1, y1), (x2, y2), VEHICLE_COLOR, max(2, image.shape[0] // 360))
