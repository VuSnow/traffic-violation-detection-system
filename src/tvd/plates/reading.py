"""Reading one vehicle's plate in one frame, as Notebook 06 does. One frame is never enough: see ``voting.py``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tvd.domain import Box
from tvd.plates.base import PAD, BasePlateModels, PlateBox
from tvd.plates.settings import ReadingSettings


@dataclass(frozen=True)
class PlateRead:
    """One frame's reading of one vehicle's plate. ``rejected`` says why it cannot be used, or is None if it can."""

    frame: int
    text: str
    mean_prob: float
    min_prob: float
    has_gap: bool
    detection_confidence: float
    plate_width: int
    plate_height: int
    plate_box: Box
    rejected: str | None
    image: np.ndarray = field(repr=False, compare=False)

    def as_json(self) -> dict[str, Any]:
        """The read without its image, with the keys of the notebooks' reads."""
        return {
            "text": self.text, "mean_prob": self.mean_prob, "min_prob": self.min_prob, "has_gap": self.has_gap, "frame": self.frame,
            "detection_confidence": self.detection_confidence, "plate_width": self.plate_width, "plate_height": self.plate_height,
            "plate_box_xyxy": self.plate_box.as_list(), "rejected": self.rejected,
        }


def select_frames(boxes: dict[int, Box], count: int, min_gap: int) -> list[int]:
    """The frames where the vehicle looks tallest, at least ``min_gap`` frames apart. ``boxes`` are by frame, in order."""
    chosen: list[int] = []
    for frame, _ in sorted(boxes.items(), key=lambda item: item[1].height, reverse=True):
        if all(abs(frame - c) >= min_gap for c in chosen):
            chosen.append(frame)
        if len(chosen) == count:
            break
    return chosen


def crop_vehicle(image: np.ndarray, box: Box, margin: float) -> tuple[np.ndarray, tuple[int, int]]:
    """The vehicle's region with a margin around its box, and where that region starts in the image."""
    margin_x, margin_y = margin * box.width, margin * box.height
    height, width = image.shape[:2]
    left, top = max(0, int(box.x1 - margin_x)), max(0, int(box.y1 - margin_y))
    right, bottom = min(width, int(box.x2 + margin_x)), min(height, int(box.y2 + margin_y))
    return image[top:bottom, left:right], (left, top)


def pick_plate(plates: list[PlateBox], box: Box, offset: tuple[int, int], lowest_center: float) -> PlateBox | None:
    """The plate most likely to be this vehicle's: inside its box, and nearest the middle of its bottom edge.

    A box can also hold a neighbour's plate, a roof sign or a shop sign, so the most confident plate is not enough.
    """
    width, height = max(box.width, 1), max(box.height, 1)
    best, best_distance = None, float("inf")
    for plate in plates:
        center_x, center_y = plate.box.center
        across = (center_x + offset[0] - box.x1) / width
        down = (center_y + offset[1] - box.y1) / height
        if not (0 <= across <= 1 and lowest_center <= down <= 1.05):
            continue
        distance = float(np.hypot(across - 0.5, down - 1.0))
        if distance < best_distance:
            best, best_distance = plate, distance
    return best


def read_text(models: BasePlateModels, plate: np.ndarray) -> tuple[str, float, float, bool]:
    """The text on a plate image, its mean and lowest character confidence, and whether an empty slot splits it.

    Confidence counts real characters only, because the OCR is also sure of its padding.
    """
    result = models.recognize(plate)
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    real = [i for i, char in enumerate(result.characters) if char != PAD]
    text = "".join(result.characters[i] for i in real)
    mean_prob = round(float(probabilities[real].mean()), 4) if real else 0.0
    min_prob = round(float(probabilities[real].min()), 4) if real else 0.0
    return text, mean_prob, min_prob, PAD in result.characters.rstrip(PAD)


def rejection_reason(text: str, min_prob: float, has_gap: bool, plate_width: int, settings: ReadingSettings) -> str | None:
    """Why a read cannot be used, or None if it can."""
    if plate_width < settings.min_plate_width:
        return "too_small"
    if len(text) < settings.min_chars:
        return "too_short"
    if has_gap:
        return "gap_in_text"
    if min_prob < settings.min_char_prob:
        return "low_confidence"
    return None


def read_vehicle(image: np.ndarray, frame: int, box: Box, models: BasePlateModels, settings: ReadingSettings) -> PlateRead | None:
    """Read one vehicle's plate in one frame. None means that no plate of this vehicle was found there."""
    crop, offset = crop_vehicle(image, box, settings.vehicle_margin)
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    plate = pick_plate(models.detect(crop), box, offset, settings.lowest_plate_center)
    if plate is None:
        return None
    b = plate.box
    plate_image = crop[max(0, int(b.y1)):max(0, int(b.y2)), max(0, int(b.x1)):max(0, int(b.x2))]
    if plate_image.size == 0:
        return None
    text, mean_prob, min_prob, has_gap = read_text(models, plate_image)
    width, height = int(plate_image.shape[1]), int(plate_image.shape[0])
    return PlateRead(
        frame=frame, text=text, mean_prob=mean_prob, min_prob=min_prob, has_gap=has_gap,
        detection_confidence=round(float(plate.confidence), 4), plate_width=width, plate_height=height,
        plate_box=Box(int(b.x1) + offset[0], int(b.y1) + offset[1], int(b.x2) + offset[0], int(b.y2) + offset[1]),
        rejected=rejection_reason(text, min_prob, has_gap, width, settings), image=plate_image.copy(),
    )
