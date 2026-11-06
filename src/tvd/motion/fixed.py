"""A camera that does not move."""

from tvd.domain import Detection
from tvd.motion.base import MOTION_ESTIMATORS, BaseMotionEstimator, CameraMotion
from tvd.video import VideoReader


@MOTION_ESTIMATORS.register("fixed")
class FixedCamera(BaseMotionEstimator):
    """Every frame is the reference frame. Nothing is computed, which saves most of a moving camera's time."""

    def estimate(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> CameraMotion:
        return CameraMotion()
