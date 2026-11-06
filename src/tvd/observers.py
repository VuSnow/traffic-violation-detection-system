"""Work that must see every frame of the pass over a video, next to detection and tracking."""

from abc import ABC, abstractmethod

from tvd.domain import Detection, Frame


class FrameObserver(ABC):
    """Sees each frame once, with all of its detections, before they are filtered for the tracker.

    Reading the signals or keeping the detections for camera-motion estimation are observers. A new
    kind of per-frame work is a new observer, and the pass over the video does not change (open/closed).
    """

    @abstractmethod
    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        """Called once per frame, in order."""


class DetectionLog(FrameObserver):
    """Every detection of every frame, of every class: what camera-motion estimation masks out of the image."""

    def __init__(self) -> None:
        self.by_frame: dict[int, list[Detection]] = {}

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self.by_frame[frame.index] = list(detections)
