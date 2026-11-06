"""What every tracker offers."""

from abc import ABC, abstractmethod

from tvd.domain import Detection, Frame, TrackedObject
from tvd.registry import Registry


class BaseTracker(ABC):
    """Links detections across frames, giving each object one track id.

    Any tracker can replace any other (the Liskov substitution principle): each takes a frame's detections
    and returns the objects it tracks in that frame, and nothing else in the pipeline knows which it is.
    """

    @abstractmethod
    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        """Advance by one frame, and return the objects tracked in it."""

    @abstractmethod
    def reset(self) -> None:
        """Forget every track, as at a cut in the video."""


TRACKERS: Registry[BaseTracker] = Registry("tracker")
