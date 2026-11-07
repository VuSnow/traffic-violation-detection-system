"""Night: the colour of the glow around a whole head."""

import cv2
import numpy as np

from tvd.signals.base import GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader


@SIGNAL_READERS.register("glow")
class GlowReader(BaseSignalReader):
    """Counts lit pixels of each colour in a box around one head (``roi``, as ``[x1, y1, x2, y2]``).

    A monochrome (infrared) image has no colour to read, and gives ``unknown``, as does a box outside the
    image. A colour wins if it has at least ``min_pixels`` and twice as many as the others together.
    """

    def __init__(self, roi: list[int], min_pixels: int = 12, min_mean_saturation: float = 8) -> None:
        self.roi = tuple(int(v) for v in roi)
        x1, y1, x2, y2 = self.roi
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"The glow box {list(self.roi)} is empty: it needs x2 > x1 and y2 > y1")
        self.min_pixels = min_pixels
        self.min_mean_saturation = min_mean_saturation

    def read(self, image: np.ndarray) -> str:
        x1, y1, x2, y2 = self.roi
        patch = image[max(0, y1):y2, max(0, x1):x2]
        if patch.size == 0:
            return UNKNOWN
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
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
