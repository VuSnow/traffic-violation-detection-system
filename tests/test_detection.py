"""Tests for traffic_violation_detection.detection, with a stand-in detector: no GPU or weights are needed."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_violation_detection import detection
from traffic_violation_detection.detection import CLASS_COLORS_BGR, DETECTOR_CONFIG, annotate_video, detect, draw_detections

VIDEO = Path(__file__).resolve().parents[1] / "data" / "traffic_sample.mp4"  # 640x360, 24 FPS, 240 frames
SHORT_VIDEO_FRAMES = 240
CAR_ID = 2


class FakeBoxes:
    def __init__(self, rows):
        rows = np.array(rows, dtype=np.float64).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = rows[:, :4], rows[:, 4], rows[:, 5]


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeDetector:
    """Returns the same boxes for every frame, and records what it was given."""

    names = {0: "person", 2: "car", 7: "truck"}

    def __init__(self, rows=()):
        self.rows = rows
        self.calls = 0
        self.kwargs = None
        self.images = []

    def predict(self, image, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.calls <= 3:
            self.images.append(image.copy())
        return [FakeResult(self.rows)]


def frames_of(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return frames, fps


BOX_ABOVE = (100, 100, 300, 250, 0.80, CAR_ID)
BOX_BELOW = (400, 100, 600, 250, 0.20, CAR_ID)


def test_output_is_named_after_the_video_and_saved_in_the_folder(tmp_path):
    path = annotate_video(VIDEO, tmp_path, model=FakeDetector())
    assert path == tmp_path / "traffic_sample_output.mp4" and path.is_file()


def test_output_folder_is_created_when_missing(tmp_path):
    folder = tmp_path / "a" / "b"
    assert annotate_video(VIDEO, folder, model=FakeDetector()).parent == folder


def test_paths_may_be_strings_and_names_may_have_spaces(tmp_path):
    source = tmp_path / "2024-04-21 10-25-27.mp4"
    source.write_bytes(VIDEO.read_bytes())
    path = annotate_video(str(source), str(tmp_path / "out put"), model=FakeDetector())
    assert path.name == "2024-04-21 10-25-27_output.mp4" and path.is_file()


def test_output_keeps_every_frame_the_size_and_the_frame_rate(tmp_path):
    model = FakeDetector()
    frames, fps = frames_of(annotate_video(VIDEO, tmp_path, model=model))
    assert len(frames) == SHORT_VIDEO_FRAMES == model.calls
    assert frames[0].shape[:2] == (360, 640) and fps == pytest.approx(24.0)


def test_only_boxes_above_the_display_threshold_are_drawn(tmp_path):
    frames, _ = frames_of(annotate_video(VIDEO, tmp_path, model=FakeDetector([BOX_ABOVE, BOX_BELOW])))
    color = np.array(CLASS_COLORS_BGR["car"], dtype=np.float64)
    distance = lambda pixels: np.abs(pixels.astype(np.float64) - color).mean()
    # The bottom edge of each box, away from the label.
    assert distance(frames[0][250, 150:250]) < 40
    assert distance(frames[0][250, 450:550]) > 40


def test_detector_sees_clean_frames_with_the_chosen_settings(tmp_path):
    model = FakeDetector([BOX_ABOVE])
    annotate_video(VIDEO, tmp_path, model=model)
    assert {k: model.kwargs[k] for k in ("imgsz", "conf", "iou", "classes")} == {k: DETECTOR_CONFIG[k] for k in ("imgsz", "conf", "iou", "classes")}
    first_source = frames_of(VIDEO)[0][0]
    assert np.array_equal(model.images[0], first_source)


def test_input_video_is_left_unchanged(tmp_path):
    before = VIDEO.read_bytes()
    annotate_video(VIDEO, tmp_path, model=FakeDetector([BOX_ABOVE]))
    assert VIDEO.read_bytes() == before


def test_missing_video_fails_before_the_detector_is_loaded(tmp_path, monkeypatch):
    def must_not_load():
        raise AssertionError("the detector was loaded")

    monkeypatch.setattr(detection, "load_detector", must_not_load)
    with pytest.raises(FileNotFoundError):
        annotate_video(tmp_path / "missing.mp4", tmp_path)


def test_detector_is_loaded_when_none_is_given(tmp_path, monkeypatch):
    model = FakeDetector()
    monkeypatch.setattr(detection, "load_detector", lambda: model)
    annotate_video(VIDEO, tmp_path)
    assert model.calls == SHORT_VIDEO_FRAMES


def test_detector_failure_leaves_no_output(tmp_path):
    class Failing(FakeDetector):
        def predict(self, image, **kwargs):
            if self.calls == 5:
                raise RuntimeError("out of memory")
            return super().predict(image, **kwargs)

    with pytest.raises(RuntimeError, match="out of memory"):
        annotate_video(VIDEO, tmp_path, model=Failing())
    assert list(tmp_path.iterdir()) == []


def test_earlier_output_is_replaced(tmp_path):
    (tmp_path / "traffic_sample_output.mp4").write_bytes(b"old")
    path = annotate_video(VIDEO, tmp_path, model=FakeDetector())
    assert path.stat().st_size > 3


def test_detect_returns_plain_detections():
    model = FakeDetector([BOX_ABOVE])
    assert detect(model, np.zeros((10, 10, 3), np.uint8)) == [
        {"class_id": CAR_ID, "class_name": "car", "confidence": 0.8, "bbox_xyxy": [100.0, 100.0, 300.0, 250.0]}
    ]


def test_detect_handles_a_frame_with_no_objects():
    assert detect(FakeDetector(), np.zeros((10, 10, 3), np.uint8)) == []


def line_rows(height, width):
    # How many pixel rows the bottom edge of a box covers.
    image = np.zeros((height, width, 3), np.uint8)
    draw_detections(image, [{"class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, height // 2, 200, height - 10]}])
    return int((image[height // 2 + 20:, 100, 1] > 0).sum())


def test_lines_get_thicker_with_the_frame():
    assert line_rows(360, 640) < line_rows(1080, 1920) < line_rows(2160, 3840)


def test_lines_at_1080p_match_notebook_02():
    # Notebook 02 drew its video with OpenCV thickness 3.
    reference = np.zeros((1080, 1920), np.uint8)
    cv2.rectangle(reference, (10, 540), (200, 1070), 255, 3)
    assert line_rows(1080, 1920) == int((reference[560:, 100] > 0).sum())


def test_unknown_class_is_drawn_in_white():
    image = np.zeros((100, 200, 3), np.uint8)
    draw_detections(image, [{"class_name": "boat", "confidence": 0.9, "bbox_xyxy": [10, 40, 150, 90]}])
    assert tuple(image[90, 100]) == (255, 255, 255)
