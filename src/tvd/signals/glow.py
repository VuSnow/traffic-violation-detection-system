"""Night: the colour of the glow around a whole head."""

import cv2
import numpy as np

from tvd.signals.base import GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader


@SIGNAL_READERS.register("glow")
class GlowReader(BaseSignalReader):
    """Counts lit pixels of each colour in a box around one head (``roi``, as ``[x1, y1, x2, y2]``).

    A monochrome (infrared) image has no colour to read, and gives ``unknown``. A colour wins if it has
    at least ``min_pixels`` and twice as many as the others together.
    """

    def __init__(self, roi: list[int], min_pixels: int = 12, min_mean_saturation: float = 8) -> None:
        self.roi = tuple(int(v) for v in roi)
        self.min_pixels = min_pixels
        self.min_mean_saturation = min_mean_saturation

    def read(self, image: np.ndarray) -> str:
        x1, y1, x2, y2 = self.roi
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if saturation.mean() < self.min_mean_saturation:
            return UNKNOWN
        lit = (value >= 120) & (saturation >= 70)
        counts = {
            RED: int((lit & ((hue <= 10) | (hue >= 165))).sum()),
            YELLOW: int((lit & (hue >= 14) & (hue <= 35)).sum()),
            GREEN: int((lit & (hue >= 40) & (hue <= 105)).sum()),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels and counts[best] >= 2 * (sum(counts.values()) - counts[best]) else UNKNOWN
