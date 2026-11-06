"""Detection and tracking over a video."""

from collections.abc import Sequence

from tvd.detection import BaseDetector
from tvd.observers import FrameObserver
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
    observers: Sequence[FrameObserver] = (),
) -> TrackHistory:
    """Detect every frame from ``start`` to ``stop``, and track the detections of ``track_classes``.

    Each observer sees every frame with all of its detections, so the video is decoded once for all of
    them. It depends only on abstractions, so any detector, tracker and observer work here.
    """
    tracker.reset()
    history = TrackHistory()
    for frame in reader.frames(start, stop):
        detections = detector.detect(frame)
        for observer in observers:
            observer.observe(frame, detections)
        history.add(frame.index, tracker.update([d for d in detections if d.class_name in track_classes], frame))
    return history
