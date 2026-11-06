"""Detection and tracking over a video."""

from tvd.detection import BaseDetector
from tvd.tracking import BaseTracker
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader


def track_video(
    reader: VideoReader,
    detector: BaseDetector,
    tracker: BaseTracker,
    track_classes: set[str],
    start: int = 0,
    stop: int | None = None,
) -> TrackHistory:
    """Detect every frame from ``start`` to ``stop``, and track the detections of ``track_classes``.

    It depends only on the two abstractions, so any detector and any tracker work here.
    """
    tracker.reset()
    history = TrackHistory()
    for frame in reader.frames(start, stop):
        detections = [d for d in detector.detect(frame) if d.class_name in track_classes]
        history.add(frame.index, tracker.update(detections, frame))
    return history
