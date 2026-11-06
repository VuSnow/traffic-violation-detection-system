"""Trackers from BoxMOT, behind ``BaseTracker``."""

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from tvd.domain import Box, Detection, Frame, TrackedObject
from tvd.tracking.base import TRACKERS, BaseTracker


def import_boxmot():
    """BoxMOT, imported only when a tracker is built, with its log messages limited to warnings."""
    try:
        import boxmot
        import boxmot.utils  # noqa: F401  Imported so that its logger exists before it is quietened.
    except ImportError as error:
        raise RuntimeError('BoxMOT trackers need BoxMOT: pip install -e ".[tracking]" && pip install --no-deps boxmot==26.0.0') from error
    logging.getLogger("boxmot").setLevel(logging.WARNING)
    return boxmot


class BoxmotTracker(BaseTracker):
    """What every BoxMOT tracker shares: converting detections to BoxMOT's array and its output back.

    A subclass only says how to build its own tracker (the template method pattern), from the options
    of its settings.
    """

    def __init__(self, **options: Any) -> None:
        self._options = options
        self._tracker = self._create(import_boxmot())

    @abstractmethod
    def _create(self, boxmot: Any) -> Any:
        """A new BoxMOT tracker, built from ``self._options``."""

    def reset(self) -> None:
        self._tracker = self._create(import_boxmot())

    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        rows = np.array([[*d.box.as_list(), d.confidence, d.class_id] for d in detections], dtype=np.float32).reshape(-1, 6)
        output = self._tracker.update(rows, frame.image)
        tracked = []
        for row in output if len(output) else []:
            matched = detections[int(row[7])]  # BoxMOT's last column is the index of the detection matched in this frame.
            box = Box(*(round(float(v), 2) for v in row[:4]))
            tracked.append(TrackedObject(int(row[4]), box, matched.class_id, matched.class_name, round(float(row[5]), 4)))
        return tracked


@TRACKERS.register("bytetrack")
class ByteTrackTracker(BoxmotTracker):
    def _create(self, boxmot: Any) -> Any:
        return boxmot.ByteTrack(boxmot.ByteTrackConfig(**self._options))


@TRACKERS.register("ocsort")
class OcSortTracker(BoxmotTracker):
    def _create(self, boxmot: Any) -> Any:
        return boxmot.OcSort(boxmot.OcSortConfig(**self._options))


@TRACKERS.register("botsort")
class BotSortTracker(BoxmotTracker):
    def _create(self, boxmot: Any) -> Any:
        return boxmot.BotSort(boxmot.BotSortConfig(**self._options))
