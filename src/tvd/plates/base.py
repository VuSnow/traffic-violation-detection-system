"""What plate reading needs from its models: to find plates in an image, and to read the characters of one."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from tvd.domain import Box
from tvd.registry import Registry

PAD = "_"
"""What an OCR model puts in a character slot that holds no character."""


@dataclass(frozen=True)
class PlateBox:
    """A plate found in an image, in that image's pixels."""

    box: Box
    confidence: float


@dataclass(frozen=True)
class PlateText:
    """What the OCR read on one plate, slot by slot: ``characters`` holds ``PAD`` in empty slots, and each slot has a confidence."""

    characters: str
    probabilities: tuple[float, ...]


class BasePlateModels(ABC):
    """A plate detector and an OCR model, loaded once and used for every vehicle. Images are BGR, as OpenCV gives them.

    Close them when done, or use them in a ``with`` block: an implementation may hold a process or a GPU.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[PlateBox]:
        """The plates in an image."""

    @abstractmethod
    def recognize(self, plate: np.ndarray) -> PlateText:
        """The characters on one plate's image."""

    def close(self) -> None:
        """Release what the models hold. Nothing, by default."""

    def is_alive(self) -> bool:
        """False once the models can no longer answer, and must be started again. Always True, by default."""
        return True

    def __enter__(self) -> BasePlateModels:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


PLATE_MODELS: Registry[BasePlateModels] = Registry("plate models")
