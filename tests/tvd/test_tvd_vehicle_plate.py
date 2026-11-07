import importlib.util

import cv2
import pytest
from conftest import ROOT, FakePlateModels
from notebook_data import NOTEBOOK_08_CLIPS, NOTEBOOK_08_PLATES, notebook_08_clips, notebook_08_plates

from tvd.domain import Box
from tvd.factory import build_plate_models
from tvd.plates import NO_PLATE, READ, load_plates_config, read_vehicle_plate, save_plate_image
from tvd.video import VideoReader

PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
READING = PLATES.reading
needs_models_and_notebook_08 = pytest.mark.skipif(
    not all(importlib.util.find_spec(n) for n in ("onnxruntime", "fast_plate_ocr", "open_image_models")) or not NOTEBOOK_08_PLATES.is_dir(),
    reason="Needs the plate models and the outputs of Notebook 08, as on the GPU server.",
)


def growing_car(frames: range) -> dict[int, Box]:
    """A car whose box grows by 1 px per frame, as one coming toward the camera."""
    return {k: Box(200, 100, 400 + k, 260 + k) for k in frames}


def test_the_frames_where_the_vehicle_is_largest_vote(sample_video):
    with VideoReader(sample_video) as video:
        result = read_vehicle_plate(video, growing_car(range(40, 100)), FakePlateModels(), READING)
    assert (result.status, result.text, result.votes) == (READ, "TEST1234", 12)
    assert [r.frame for r in result.reads] == list(range(55, 100, 4))  # The 12 tallest boxes, 4 frames apart, in the order of the video.


def test_a_vehicle_never_tracked_has_no_plate(sample_video):
    with VideoReader(sample_video) as video:
        assert read_vehicle_plate(video, {}, FakePlateModels(), READING).status == NO_PLATE


def test_the_best_read_is_saved_as_a_png(tmp_path, sample_video):
    with VideoReader(sample_video) as video:
        result = read_vehicle_plate(video, growing_car(range(40, 100)), FakePlateModels(), READING)
    path = save_plate_image(result, tmp_path / "plates", "evt-1")
    assert path.name == "evt-1_f00099.png" and cv2.imread(str(path)).shape == (result.best_read().plate_height, result.best_read().plate_width, 3)


def test_no_read_saves_nothing(tmp_path, sample_video):
    with VideoReader(sample_video) as video:
        assert save_plate_image(read_vehicle_plate(video, {}, FakePlateModels(), READING), tmp_path, "evt-1") is None


@needs_models_and_notebook_08
def test_the_same_plates_and_images_as_notebook_08(tmp_path):
    saved = notebook_08_plates()
    with build_plate_models(PLATES.models) as models:
        for entry in notebook_08_clips():
            boxes = {b["frame_in_clip"]: Box(*b["box_xyxy"]) for b in entry["vehicle_boxes"]}
            with VideoReader(NOTEBOOK_08_CLIPS / entry["clip_file"]) as clip:
                result = read_vehicle_plate(clip, boxes, models, READING)
            path = save_plate_image(result, tmp_path, entry["event_id"])
            expected = saved[entry["event_id"]]
            same_result = result.as_json() == expected["license_plate"]  # Outside the assert, so that a failure never prints a real plate.
            assert same_result and result.status == READ, entry["event_id"]
            assert path.name == expected["plate_crop"] and path.read_bytes() == (NOTEBOOK_08_PLATES / "plate_crops" / expected["plate_crop"]).read_bytes()
