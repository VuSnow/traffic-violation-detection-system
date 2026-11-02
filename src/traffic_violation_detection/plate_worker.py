"""Runs the plate models in a process of their own, on the GPU when there is one.

ONNX Runtime and PyTorch disturb each other inside one process: after ONNX had run on the GPU,
PyTorch's detections moved by about 0.01 px, enough to change 15 of 136 tracks, so a video's result
depended on what had run before it. With ONNX in another process, the detections stay identical
(Notebook 06). Why it happens inside one process is not known.

This file holds both sides. ``PlateWorker`` lives in the caller's process, and ``main`` in the worker's.
``PlateDetector`` and ``PlateReader`` offer the calls that ``plates.py`` makes on the real models.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

STOP_TIMEOUT_SECONDS = 10


def create_models(settings: dict) -> tuple:
    """The real models, for the worker's process only: importing ONNX Runtime anywhere else would defeat its purpose."""
    try:
        import onnxruntime as ort
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector
    except ImportError as error:
        raise RuntimeError('Plate reading needs its models: pip install -e ".[plates]"') from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings["cpu_threads"]
    options.inter_op_num_threads = 1
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    arguments = {"providers": providers, "sess_options": options}
    return create_detector(settings["detector_model"], **arguments), LicensePlateRecognizer(settings["ocr_model"], **arguments)


def detect_plates(detector, image) -> list[SimpleNamespace]:
    """Detected plates as plain values, which can cross between processes."""
    return [
        SimpleNamespace(
            confidence=float(d.confidence),
            bounding_box=SimpleNamespace(x1=d.bounding_box.x1, y1=d.bounding_box.y1, x2=d.bounding_box.x2, y2=d.bounding_box.y2),
        )
        for d in detector.predict(image)
    ]


def read_plate(reader, image, return_confidence: bool, remove_pad_char: bool) -> SimpleNamespace:
    """An OCR prediction as plain values, which can cross between processes."""
    prediction = reader.run_one(image, return_confidence=return_confidence, remove_pad_char=remove_pad_char)
    return SimpleNamespace(plate=prediction.plate, char_probs=prediction.char_probs)


def main() -> None:
    """The worker's process: answer requests from standard input until the caller closes it."""
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # What the libraries print must not reach the replies.

    def reply(ok: bool, value) -> None:
        pickle.dump((ok, value), replies, protocol=pickle.HIGHEST_PROTOCOL)
        replies.flush()

    try:
        detector, reader = create_models(json.loads(sys.argv[1]))
    except Exception as error:
        reply(False, f"{type(error).__name__}: {error}")
        return
    reply(True, detector.model.get_providers())
    methods = {"detect": lambda *a: detect_plates(detector, *a), "read": lambda *a: read_plate(reader, *a)}
    while True:
        try:
            method, arguments = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            reply(True, methods[method](*arguments))
        except Exception as error:
            reply(False, f"{type(error).__name__}: {error}")


def stop_process(process: subprocess.Popen) -> None:
    """Close the worker's input so that it ends by itself, and force it if it does not."""
    if process.poll() is None:
        try:
            process.stdin.close()
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()


class PlateWorker:
    """The caller's side of a worker process. It stops when closed, or when nothing refers to it any more."""

    def __init__(self, settings: dict):
        source_root = str(Path(__file__).resolve().parents[1])
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")]))}
        self._process = subprocess.Popen(
            [sys.executable, "-m", "traffic_violation_detection.plate_worker", json.dumps(settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment,
        )
        self._finalizer = weakref.finalize(self, stop_process, self._process)
        self.providers = self._receive()

    @property
    def pid(self) -> int:
        return self._process.pid

    def _receive(self):
        try:
            ok, value = pickle.load(self._process.stdout)
        except EOFError as error:
            raise RuntimeError(f"The plate worker stopped (exit code {self._process.wait()}); its messages are on standard error.") from error
        if not ok:
            raise RuntimeError(f"Plate worker: {value}")
        return value

    def call(self, method: str, *arguments):
        """Run a method in the worker and return its result. An error there is raised here."""
        try:
            pickle.dump((method, arguments), self._process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise RuntimeError(f"The plate worker stopped (exit code {self._process.poll()}).") from error
        return self._receive()

    def close(self) -> None:
        self._finalizer()


class PlateDetector:
    """The worker's plate detector, with the call that ``plates.py`` makes on the real one."""

    def __init__(self, worker: PlateWorker):
        self.worker = worker

    def predict(self, image) -> list[SimpleNamespace]:
        return self.worker.call("detect", image)


class PlateReader:
    """The worker's OCR, with the call that ``plates.py`` makes on the real one."""

    def __init__(self, worker: PlateWorker):
        self.worker = worker

    def run_one(self, image, return_confidence: bool = False, remove_pad_char: bool = True) -> SimpleNamespace:
        return self.worker.call("read", image, return_confidence, remove_pad_char)


if __name__ == "__main__":
    main()
