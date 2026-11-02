"""Tests for traffic_violation_detection.plate_worker, with the real models in a real process."""

import gc
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from traffic_violation_detection.plate_worker import PlateDetector, PlateReader, PlateWorker
from traffic_violation_detection.plates import PLATE_SETTINGS

pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("onnxruntime", "fast_plate_ocr", "open_image_models")),
    reason="The plate models are not installed here.",
)
GRAY = np.full((60, 120, 3), 128, np.uint8)


@pytest.fixture(scope="module")
def worker():
    started = PlateWorker(PLATE_SETTINGS)
    yield started
    started.close()


def test_worker_is_another_process(worker):
    assert worker.pid != os.getpid()


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="There is no GPU here.")
def test_worker_uses_the_gpu_where_there_is_one(worker):
    assert worker.providers[0] == "CUDAExecutionProvider"


def test_detector_answers_like_the_real_one(worker):
    assert PlateDetector(worker).predict(GRAY) == []


def test_reader_answers_like_the_real_one(worker):
    prediction = PlateReader(worker).run_one(GRAY, return_confidence=True, remove_pad_char=False)
    assert len(prediction.plate) == len(prediction.char_probs) == 10


def test_reader_can_remove_the_padding(worker):
    assert PlateReader(worker).run_one(GRAY, return_confidence=True).plate == ""


def test_large_images_cross_between_processes(worker):
    assert isinstance(PlateDetector(worker).predict(np.zeros((1080, 1920, 3), np.uint8)), list)


def test_error_in_the_worker_is_raised_here_and_the_worker_survives(worker):
    with pytest.raises(RuntimeError, match="Plate worker: "):
        worker.call("detect", "not an image")
    assert PlateDetector(worker).predict(GRAY) == []


def test_unknown_method_is_an_error_not_a_hang(worker):
    with pytest.raises(RuntimeError, match="unknown"):
        worker.call("unknown")


def test_unknown_model_fails_when_the_worker_starts():
    with pytest.raises(RuntimeError, match="Plate worker: "):
        PlateWorker({**PLATE_SETTINGS, "ocr_model": "no-such-model"})


def test_closing_stops_the_process():
    started = PlateWorker(PLATE_SETTINGS)
    process = started._process
    started.close()
    assert process.poll() is not None


def test_process_stops_when_nothing_refers_to_the_worker():
    started = PlateWorker(PLATE_SETTINGS)
    process = started._process
    del started
    gc.collect()
    assert process.wait(timeout=15) == 0


def test_dead_worker_gives_a_clear_error():
    started = PlateWorker(PLATE_SETTINGS)
    started._process.kill()
    started._process.wait()
    with pytest.raises(RuntimeError, match="plate worker stopped"):
        started.call("detect", GRAY)


def test_onnx_stays_out_of_the_callers_process():
    # In one process with PyTorch it changes the detections, and with them the tracks.
    code = (
        "import sys\n"
        "from traffic_violation_detection.plates import load_plate_models\n"
        "detector, reader = load_plate_models()\n"
        "assert detector.predict(__import__('numpy').zeros((60, 120, 3), 'uint8')) == []\n"
        "assert 'onnxruntime' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
