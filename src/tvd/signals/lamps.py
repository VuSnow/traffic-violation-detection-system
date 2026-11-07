"""Day: the colour at each lamp's position."""

import cv2
import numpy as np

from tvd.signals.base import GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader


@SIGNAL_READERS.register("lamps")
class LampReader(BaseSignalReader):
    """Counts lit pixels in a small square around each lamp of one head.

    Each lamp is read separately, because a head's yellow backplate confuses a count over the whole
    head. A colour wins if it has at least ``min_pixels`` and twice as many as the others together.
    """

    def __init__(self, red: list[int], yellow: list[int], green: list[int], half: int = 6, min_pixels: int = 6) -> None:
        self.lamps = {RED: tuple(red), YELLOW: tuple(yellow), GREEN: tuple(green)}
        self.half = half
        self.min_pixels = min_pixels

    def read(self, image: np.ndarray) -> str:
        counts = {}
        height, width = image.shape[:2]
        half = self.half
        for colour, (x, y) in self.lamps.items():
            if not (half <= x < width - half and half <= y < height - half):
                counts[colour] = 0
                continue
            hsv = cv2.cvtColor(image[y - half:y + half + 1, x - half:x + half + 1], cv2.COLOR_BGR2HSV)
            counts[colour] = int(lit_pixels(colour, hsv[..., 0], hsv[..., 1], hsv[..., 2]).sum())
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels and counts[best] >= 2 * (sum(counts.values()) - counts[best]) else UNKNOWN


def lit_pixels(colour: str, hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Pixels lit in ``colour``, with the HSV thresholds found in Notebook 05b. OpenCV's hue runs from 0 to 179."""
    if colour == RED:
        return ((hue <= 10) | (hue >= 165)) & (saturation >= 90) & (value >= 90)
    if colour == YELLOW:
        return (hue >= 14) & (hue <= 35) & (saturation >= 120) & (value >= 170)
    return (hue >= 40) & (hue <= 105) & (saturation >= 60) & (value >= 110)
