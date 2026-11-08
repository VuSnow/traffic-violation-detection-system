"""The real plate models, in a process of their own: ``onnx_worker``.

ONNX Runtime and PyTorch disturb each other inside one process: after ONNX had run on the GPU, PyTorch's
detections moved by about 0.01 px, enough to change 15 of 136 tracks, so a video's result depended on
what had run before it (Notebook 06). With ONNX in another process, the detections stay the same.
Why it happens inside one process is not known.

This class is the caller's side; ``onnx_process.py`` is the other process.
"""

from __future__ import annotations

import json
import os
import pickle
import select
import subprocess
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from tvd.plates.base import PLATE_MODELS, BasePlateModels, PlateBox, PlateText

STOP_TIMEOUT_SECONDS = 10


def stop_process(process: subprocess.Popen) -> None:
    """Close the process's input, so that it ends by itself, and force it if it does not."""
    if process.poll() is None:
        try:
            process.stdin.close()
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()


@PLATE_MODELS.register("onnx_worker")
class OnnxPlateModels(BasePlateModels):
    """Starts the models' process, and asks it to detect and read. It stops when closed, or when nothing refers to it any more."""

    def __init__(self, detector_model: str, ocr_model: str, cpu_threads: int, timeout_seconds: float) -> None:
        options = {"detector_model": detector_model, "ocr_model": ocr_model, "cpu_threads": cpu_threads}
        self.timeout_seconds = timeout_seconds
        source_root = str(Path(__file__).resolve().parents[2])  # So that the process runs this same code.
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")]))}
        self._process = subprocess.Popen(
            [sys.executable, "-m", "tvd.plates.onnx_process", json.dumps(options)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment,
        )
        self._finalizer = weakref.finalize(self, stop_process, self._process)
        self.providers: list[str] = self._receive()  # Loading the models may take long, the first time: no time limit.

    @property
    def pid(self) -> int:
        return self._process.pid

    def _receive(self, timeout: float | None = None) -> Any:
        if timeout is not None and not select.select([self._process.stdout], [], [], timeout)[0]:
            self._process.kill()
            self._process.wait()
            raise RuntimeError(f"The plate models did not answer within {timeout:g} s; their process was stopped")
        try:
            ok, value = pickle.load(self._process.stdout)
        except EOFError as error:
            raise RuntimeError(f"The plate models' process stopped (exit code {self._process.wait()}); its messages are on standard error.") from error
        if not ok:
            raise RuntimeError(f"Plate models: {value}")
        return value

    def call(self, method: str, *arguments: Any) -> Any:
        """Run a method in the process and return its answer. An error there is raised here, and the process goes on."""
        try:
            pickle.dump((method, arguments), self._process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise RuntimeError(f"The plate models' process stopped (exit code {self._process.poll()}).") from error
        return self._receive(self.timeout_seconds)

    def detect(self, image: np.ndarray) -> list[PlateBox]:
        return self.call("detect", image)

    def recognize(self, plate: np.ndarray) -> PlateText:
        return self.call("recognize", plate)

    def close(self) -> None:
        self._finalizer()

    def is_alive(self) -> bool:
        return self._process.poll() is None
