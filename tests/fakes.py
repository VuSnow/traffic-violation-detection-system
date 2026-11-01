"""Stand-ins for the detector and tracker, and small videos, so that tests need no GPU or weights."""

import numpy as np

from traffic_violation_detection.video import video_writer


class FakeBoxes:
    def __init__(self, rows):
        rows = np.array(rows, dtype=np.float64).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = rows[:, :4], rows[:, 4], rows[:, 5]


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class ScriptedDetector:
    """``script(call_number)`` gives the rows ``x1, y1, x2, y2, confidence, class_id`` of each frame it is shown."""

    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light"}

    def __init__(self, script=lambda call: []):
        self.script = script
        self.calls = 0

    def predict(self, image, **kwargs):
        rows = self.script(self.calls)
        self.calls += 1
        return [FakeResult(rows)]


class PassThroughTracker:
    """Tracks each detection under an id equal to its row number, in the output format of BoxMOT."""

    def update(self, detections, image):
        return np.array([[*d[:4], i + 1, d[4], d[5], i] for i, d in enumerate(detections)], dtype=np.float64).reshape(-1, 8)


def write_video(path, frames, fps=30.0, crf=14):
    with video_writer(path, fps, crf) as write_frame:
        for frame in frames:
            write_frame(frame)
    return path
