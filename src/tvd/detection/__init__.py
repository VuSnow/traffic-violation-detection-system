"""Detectors. Importing this package registers every detector in it."""

from tvd.detection import yolo  # noqa: F401  Registers "yolo".
from tvd.detection.base import DETECTORS, BaseDetector

__all__ = ["DETECTORS", "BaseDetector"]
