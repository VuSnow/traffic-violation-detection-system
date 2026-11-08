"""What every detector offers."""

from abc import ABC, abstractmethod
from typing import Any

from tvd.domain import Detection, Frame
from tvd.registry import Registry


class BaseDetector(ABC):
    """Finds objects in one frame.

    Every detector returns the same ``Detection`` type, so nothing after it depends on a model library.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """The objects found in ``frame``."""

    def describe(self) -> dict[str, Any]:
        """What this detector is, for records and the health check. A model's detector names its weights and device."""
        return {"type": type(self).__name__}


DETECTORS: Registry[BaseDetector] = Registry("detector")
