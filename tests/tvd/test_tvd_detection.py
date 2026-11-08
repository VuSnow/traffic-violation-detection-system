import importlib.util

import numpy as np
import pytest
from conftest import ROOT

from tvd.config import file_version
from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Frame
from tvd.video import VideoReader

needs_yolo = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not (ROOT / "weights" / "yolo26m.pt").is_file(),
    reason="Needs Ultralytics and weights/yolo26m.pt, as on the GPU server.",
)
NOTEBOOK_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]


def test_a_detector_must_implement_detect():
    class Incomplete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_yolo_is_registered():
    assert "yolo" in DETECTORS.names()


def test_the_test_double_finds_a_moving_car():
    car = DETECTORS.create("test-moving-car", step=2).detect(Frame(3, 0.1, np.zeros((1, 1, 3), np.uint8)))[0]
    assert car.class_name == "car" and car.box.x1 == 16


@needs_yolo
def test_yolo_finds_the_same_boxes_as_the_notebooks(monkeypatch):
    from traffic_violation_detection.detection import DETECTOR_CONFIG, detect, load_detector

    monkeypatch.chdir(ROOT)
    notebook_model = load_detector()
    detector = DETECTORS.create("yolo", weights="yolo26m.pt", classes=NOTEBOOK_CLASSES)
    with VideoReader(ROOT / "data" / "video-test.mp4") as reader:
        for frame in reader.frames(0, 5):
            expected = [(d["class_id"], d["bbox_xyxy"], d["confidence"]) for d in detect(notebook_model, frame.image, DETECTOR_CONFIG)]
            assert [(d.class_id, d.box.as_list(), d.confidence) for d in detector.detect(frame)] == expected


@needs_yolo
def test_an_unknown_class_is_refused(monkeypatch):
    monkeypatch.chdir(ROOT)
    with pytest.raises(ValueError, match="no classes named \\['tractor'\\]"):
        DETECTORS.create("yolo", weights="yolo26m.pt", classes=["car", "tractor"])


def test_a_detector_describes_itself_by_its_class_by_default():
    assert DETECTORS.create("test-moving-car").describe() == {"type": "MovingCarDetector"}


@needs_yolo
def test_yolo_names_its_weights_their_version_and_its_device(monkeypatch):
    import torch

    monkeypatch.chdir(ROOT)
    assert DETECTORS.create("yolo", weights="yolo26m.pt").describe() == {
        "type": "yolo", "weights": "yolo26m.pt", "weights_version": file_version(ROOT / "weights" / "yolo26m.pt"),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
