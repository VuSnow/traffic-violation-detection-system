from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, FakePlateModels

from tvd.config import ConfigError
from tvd.domain import Box
from tvd.plates import PLATE_MODELS, BasePlateModels, PlateBox, load_plates_config, read_vehicle, select_frames
from tvd.plates.reading import pick_plate

PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
READING = PLATES.reading
IMAGE = np.zeros((360, 640, 3), np.uint8)
CAR = Box(200, 100, 400, 260)


class NotebookModels:
    """Models with the calls that the notebooks' plate code makes, answering as the given tvd models do."""

    def __init__(self, models: BasePlateModels) -> None:
        self.models = models

    def predict(self, image):
        return [SimpleNamespace(confidence=p.confidence, bounding_box=SimpleNamespace(x1=p.box.x1, y1=p.box.y1, x2=p.box.x2, y2=p.box.y2)) for p in self.models.detect(image)]

    def run_one(self, image, return_confidence=False, remove_pad_char=True):
        text = self.models.recognize(image)
        return SimpleNamespace(plate=text.characters, char_probs=np.array(text.probabilities))


def test_the_settings_file_has_the_notebooks_values():
    from traffic_violation_detection.plates import PLATE_SETTINGS

    options = {k: v for k, v in PLATES.models.options.items() if k != "timeout_seconds"}  # New in the service: the notebooks wait for ever.
    assert {"version": PLATES.version, **options, **READING.model_dump()} == PLATE_SETTINGS and PLATES.models.type == "onnx_worker"
    assert PLATES.models.options["timeout_seconds"] == 60


def test_a_wrong_setting_is_named(tmp_path):
    path = tmp_path / "plates.yaml"
    path.write_text((ROOT / "configs" / "tvd" / "plates.yaml").read_text().replace("min_agreement: 0.67", "min_agreement: 67"))
    with pytest.raises(ConfigError, match="reading.min_agreement: Input should be less than or equal to 1"):
        load_plates_config(path)


def test_models_are_chosen_by_name_and_closed_after_use():
    with PLATE_MODELS.create("test-plate", text="FAKE9999") as models:
        assert models.recognize(IMAGE).characters == "FAKE9999_"
    assert models.closed


def test_plate_models_describe_themselves_by_their_class_by_default():
    assert FakePlateModels().describe() == {"type": "FakePlateModels"}


def test_models_without_a_process_are_always_alive():
    assert FakePlateModels().is_alive()


def test_plate_models_must_detect_and_recognize():
    class DetectorOnly(BasePlateModels):
        def detect(self, image):
            return []

    with pytest.raises(TypeError):
        DetectorOnly()


def test_a_plate_is_read_where_the_vehicle_is():
    read = read_vehicle(IMAGE, 7, CAR, FakePlateModels(), READING)
    assert (read.frame, read.text, read.mean_prob, read.min_prob, read.has_gap, read.rejected) == (7, "TEST1234", 0.99, 0.99, False, None)
    assert read.plate_box == Box(267, 242, 332, 263) and (read.plate_width, read.plate_height) == (65, 21) and read.image.shape == (21, 65, 3)


@pytest.mark.parametrize("models, reason", [
    (FakePlateModels(prob=0.9), "low_confidence"),
    (FakePlateModels(text="TES"), "too_short"),
    (FakePlateModels(text="TES_T1234"), "gap_in_text"),
])
def test_a_read_that_cannot_be_used_says_why(models, reason):
    assert read_vehicle(IMAGE, 0, CAR, models, READING).rejected == reason


def test_a_plate_narrower_than_30_px_is_too_small():
    assert read_vehicle(IMAGE, 0, Box(300, 200, 380, 260), FakePlateModels(), READING).rejected == "too_small"


def test_a_sign_high_on_the_vehicle_is_not_its_plate():
    class RoofSign(FakePlateModels):
        def detect(self, image):
            height, width = image.shape[:2]
            return [PlateBox(Box(0.4 * width, 0.1 * height, 0.6 * width, 0.2 * height), 0.99)]

    assert read_vehicle(IMAGE, 0, CAR, RoofSign(), READING) is None


def test_the_plate_nearest_the_middle_of_the_bottom_edge_wins():
    middle, side = PlateBox(Box(280, 240, 320, 260), 0.5), PlateBox(Box(210, 240, 250, 260), 0.99)
    assert pick_plate([side, middle], CAR, (0, 0), 0.4) is middle


def test_the_frames_where_the_vehicle_is_tallest_and_apart():
    boxes = {k: Box(0, 0, 10, height) for k, height in enumerate([10, 50, 49, 48, 20, 47, 46])}
    assert select_frames(boxes, 3, 2) == [1, 3, 5]


@pytest.mark.parametrize("text, prob", [("TEST1234", 0.99), ("TES", 0.99), ("TES_T1234", 0.97), ("FAKE9999", 0.9)])
@pytest.mark.parametrize("box", [Box(200, 100, 400, 260), Box(300, 200, 380, 260), Box(-5, 300, 120, 359.5), Box(600, 0, 640, 30)])
def test_the_same_reads_as_the_notebooks(text, prob, box):
    from traffic_violation_detection.plates import PLATE_SETTINGS, read_vehicle_in_frame

    image = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    models = FakePlateModels(text, prob)
    ours = read_vehicle(image, 3, box, models, READING)
    theirs = read_vehicle_in_frame(image, [3, *box.as_list()], NotebookModels(models), NotebookModels(models), PLATE_SETTINGS)
    assert (ours is None) == (theirs is None)
    if ours is not None:
        assert ours.as_json() == {k: v for k, v in theirs.items() if k != "plate_image"} and np.array_equal(ours.image, theirs["plate_image"])
