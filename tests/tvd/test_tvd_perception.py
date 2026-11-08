import importlib.util

import pytest
from conftest import ROOT

from tvd.detection import DETECTORS
from tvd.observers import DetectionLog, FrameCounter, FrameObserver
from tvd.perception import track_video
from tvd.tracking import TRACKERS
from tvd.video import VideoReader

needs_gpu_stack = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("ultralytics", "boxmot")) or not (ROOT / "weights" / "yolo26m.pt").is_file(),
    reason="Needs Ultralytics, BoxMOT and weights/yolo26m.pt, as on the GPU server.",
)


def test_only_the_chosen_classes_are_tracked(sample_video):
    with VideoReader(sample_video) as reader:
        history = track_video(reader, DETECTORS.create("test-moving-car"), TRACKERS.create("test-same-id"), {"car"}, 0, 5)
    assert history.frames_seen == 5 and list(history.tracks) == [1]
    assert history.tracks[1].class_name == "car" and len(history.tracks[1].points) == 5


def test_the_tracker_is_reset_before_each_video(sample_video):
    tracker = TRACKERS.create("test-same-id")
    with VideoReader(sample_video) as reader:
        track_video(reader, DETECTORS.create("test-moving-car"), tracker, {"car"}, 0, 2)
        track_video(reader, DETECTORS.create("test-moving-car"), tracker, {"car"}, 0, 2)
    assert tracker.resets == 2


def test_observers_see_every_frame_with_all_its_detections(sample_video):
    class FrameCounter(FrameObserver):
        def __init__(self):
            self.frames = []

        def observe(self, frame, detections):
            self.frames.append(frame.index)

    counter, log = FrameCounter(), DetectionLog()
    with VideoReader(sample_video) as reader:
        track_video(reader, DETECTORS.create("test-moving-car"), TRACKERS.create("test-same-id"), {"car"}, 3, 6, observers=[counter, log])
    assert counter.frames == [3, 4, 5] and sorted(log.by_frame) == [3, 4, 5]
    assert [d.class_name for d in log.by_frame[4]] == ["car", "person"]


@needs_gpu_stack
def test_the_same_tracks_as_the_notebooks(monkeypatch):
    from traffic_violation_detection.detection import load_detector
    from traffic_violation_detection.tracking import track_frames

    monkeypatch.chdir(ROOT)
    video, stop = ROOT / "data" / "video-test.mp4", 150
    expected = {
        record["frame_index"]: sorted((t["track_id"], t["bbox_xyxy"]) for t in tracked)
        for record, _, tracked in track_frames(video, load_detector(), 0, stop)
    }
    detector = DETECTORS.create("yolo", weights="yolo26m.pt", classes=["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"])
    with VideoReader(video) as reader:
        history = track_video(reader, detector, TRACKERS.create("ocsort", use_byte=True), {"bicycle", "car", "motorcycle", "bus", "truck"}, 0, stop)
    found = {frame: [] for frame in expected}
    for track in history.tracks.values():
        for point in track.points:
            found[point.frame_index].append((track.track_id, point.box.as_list()))
    assert {frame: sorted(items) for frame, items in found.items()} == expected


def test_a_frame_counter_reports_every_n_frames_and_the_last(sample_video):
    reported = []
    counter = FrameCounter(reported.append, every=100, total=240)
    with VideoReader(sample_video) as reader:
        track_video(reader, DETECTORS.create("test-moving-car"), TRACKERS.create("test-same-id"), {"car"}, observers=[counter])
    assert reported == [100, 200, 240] and counter.count == 240
