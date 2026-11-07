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


def draw_label(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Lines of text, each in its own colour, on a dark band at the top left of the image."""
    scale = max(0.5, 0.8 * image.shape[0] / 1080)
    thickness = max(1, round(scale * 2.5))
    sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for text, _ in lines]
    step = max(height for _, height in sizes) + 12
    cv2.rectangle(image, (0, 0), (max(width for width, _ in sizes) + 16, step * len(lines) + 4), (0, 0, 0), -1)
    for k, ((text, color), (_, height)) in enumerate(zip(lines, sizes)):
        cv2.putText(image, text, (8, step * k + height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
