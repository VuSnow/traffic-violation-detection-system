"""What every detector offers."""

from abc import ABC, abstractmethod

from tvd.domain import Detection, Frame
from tvd.registry import Registry


class BaseDetector(ABC):
    """Finds objects in one frame.

    Every detector returns the same ``Detection`` type, so nothing after it depends on a model library.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """The objects found in ``frame``."""


DETECTORS: Registry[BaseDetector] = Registry("detector")
