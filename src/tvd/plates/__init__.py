"""Plate reading: models behind an interface, the reading of one frame, and the vote over many frames.

Importing this package registers every implementation of the models in it.
"""

from tvd.plates import onnx  # noqa: F401  Registers "onnx_worker".
from tvd.plates.base import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.plates.reading import PlateRead, read_vehicle, select_frames
from tvd.plates.settings import PlatesConfig, ReadingSettings, load_plates_config
from tvd.plates.voting import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateResult, vote

__all__ = [
    "NO_PLATE", "PAD", "PLATE_MODELS", "READ", "UNCERTAIN", "UNREADABLE", "BasePlateModels", "PlateBox", "PlateRead", "PlateResult", "PlateText",
    "PlatesConfig", "ReadingSettings", "load_plates_config", "read_vehicle", "select_frames", "vote",
]
