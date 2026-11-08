"""The real plate models, in their own process. These tests skip where the models are not installed."""

import gc
import importlib.util
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_CLIPS, NOTEBOOK_08_PLATES

from tvd.factory import build_plate_models
from tvd.plates import load_plates_config
from tvd.video import VideoReader

PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
GRAY = np.full((60, 120, 3), 128, np.uint8)
pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("onnxruntime", "fast_plate_ocr", "open_image_models")), reason="The plate models are not installed here.",
)
needs_notebook_08 = pytest.mark.skipif(not NOTEBOOK_08_PLATES.is_dir(), reason="Needs the outputs of Notebook 08, as on the GPU server.")


@pytest.fixture(scope="module")
def models():
    with build_plate_models(PLATES.models) as started:
        yield started


def test_the_models_run_in_another_process(models):
    assert models.pid != os.getpid()


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="There is no GPU here.")
def test_the_models_use_the_gpu_where_there_is_one(models):
    assert models.providers[0] == "CUDAExecutionProvider"


def test_a_gray_image_has_no_plate_and_ten_empty_slots(models):
    text = models.recognize(GRAY)
    assert models.detect(GRAY) == [] and text.characters == "_" * 10 and len(text.probabilities) == 10


def test_a_large_image_crosses_between_the_processes(models):
    assert models.detect(np.zeros((1080, 1920, 3), np.uint8)) == []


def test_an_error_there_is_raised_here_and_the_models_go_on(models):
    with pytest.raises(RuntimeError, match="Plate models: "):
        models.call("detect", "not an image")
    with pytest.raises(RuntimeError, match="KeyError"):
        models.call("unknown")
    assert models.detect(GRAY) == []


def test_an_unknown_model_fails_when_the_process_starts():
    with pytest.raises(RuntimeError, match="Plate models: "):
        build_plate_models(PLATES.models.model_copy(update={"options": {**PLATES.models.options, "ocr_model": "no-such-model"}}))


def test_closing_stops_the_process():
    started = build_plate_models(PLATES.models)
    process = started._process
    started.close()
    assert process.poll() is not None


def test_the_process_stops_when_nothing_refers_to_the_models():
    started = build_plate_models(PLATES.models)
    process = started._process
    del started
    gc.collect()
    assert process.wait(timeout=15) == 0


def test_models_that_do_not_answer_in_time_are_stopped():
    hasty = build_plate_models(PLATES.models.model_copy(update={"options": {**PLATES.models.options, "timeout_seconds": 0.0001}}))
    try:
        assert hasty.is_alive()
        with pytest.raises(RuntimeError, match="did not answer within 0.0001 s"):
            hasty.detect(np.zeros((1080, 1920, 3), np.uint8))
        assert not hasty.is_alive()
    finally:
        hasty.close()


def test_closed_models_are_not_alive():
    started = build_plate_models(PLATES.models)
    started.close()
    assert not started.is_alive()


def test_a_stopped_process_gives_a_clear_error():
    started = build_plate_models(PLATES.models)
    started._process.kill()
    started._process.wait()
    with pytest.raises(RuntimeError, match="process stopped"):
        started.detect(GRAY)


def test_onnx_stays_out_of_the_callers_process():
    # In one process with PyTorch it changes the detections, and with them the tracks.
    code = (
        "import sys\n"
        "from tvd.factory import build_plate_models\n"
        "from tvd.plates import load_plates_config\n"
        "models = build_plate_models(load_plates_config('configs/tvd/plates.yaml').models)\n"
        "assert models.detect(__import__('numpy').zeros((60, 120, 3), 'uint8')) == []\n"
        "assert 'onnxruntime' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, cwd=ROOT)


@needs_notebook_08
def test_the_same_answers_as_the_notebooks_models(models):
    from traffic_violation_detection.plate_worker import PlateDetector, PlateReader, PlateWorker
    from traffic_violation_detection.plates import PLATE_SETTINGS

    with VideoReader(next(NOTEBOOK_08_CLIPS.glob("*.mp4"))) as clip:
        frames = [f.image for f in clip.frames()][::12]
    crops = [cv2.imread(str(p)) for p in sorted((NOTEBOOK_08_PLATES / "plate_crops").glob("*.png"))]
    worker = PlateWorker(PLATE_SETTINGS)
    try:
        detector, reader = PlateDetector(worker), PlateReader(worker)
        ours = [[(p.box.as_list(), p.confidence) for p in models.detect(image)] for image in frames]
        theirs = [[([d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2], d.confidence) for d in detector.predict(image)] for image in frames]
        our_texts = [models.recognize(crop) for crop in crops]
        their_texts = [reader.run_one(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_confidence=True, remove_pad_char=False) for crop in crops]
    finally:
        worker.close()
    assert ours == theirs and sum(map(len, ours)) > 0
    same_texts = [(t.characters, t.probabilities) for t in our_texts] == [(t.plate, tuple(float(p) for p in t.char_probs)) for t in their_texts]
    assert same_texts and len(crops) == 2  # Compared outside the assert, so that a failure never prints a real plate.
