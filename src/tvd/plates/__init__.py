"""Plate reading: models behind an interface, the reading of one frame, and the vote over many frames."""

from tvd.plates.base import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.plates.reading import PlateRead, read_vehicle, select_frames
from tvd.plates.settings import PlatesConfig, ReadingSettings, load_plates_config

__all__ = [
    "PAD", "PLATE_MODELS", "BasePlateModels", "PlateBox", "PlateRead", "PlateText", "PlatesConfig", "ReadingSettings",
    "load_plates_config", "read_vehicle", "select_frames",
]
