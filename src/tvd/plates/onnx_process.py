"""The process that runs the real plate models: ``python -m tvd.plates.onnx_process '<options as JSON>'``.

It answers requests from its standard input until the caller closes it. Requests and answers are pickled
``(method, arguments)`` and ``(ok, value)``. Only this process ever imports ONNX Runtime: see ``onnx.py``.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any

import cv2

from tvd.domain import Box
from tvd.plates.base import PlateBox, PlateText


def create_models(detector_model: str, ocr_model: str, cpu_threads: int) -> tuple[Any, Any]:
    """The plate detector of open-image-models and the OCR of fast-plate-ocr, on the GPU when there is one."""
    try:
        import onnxruntime as ort
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector
    except ImportError as error:
        raise RuntimeError('Plate reading needs its models: pip install -e ".[plates]"') from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
        # ONNX Runtime needs the CUDA libraries that pip installs with PyTorch, which are not on the loader's path.
        # Without them, on a machine whose system CUDA is another version, it runs on the CPU with only a warning.
        ort.preload_dlls()
    arguments = {"providers": providers, "sess_options": options}
    return create_detector(detector_model, **arguments), LicensePlateRecognizer(ocr_model, **arguments)


def detect(detector: Any, image) -> list[PlateBox]:
    return [PlateBox(Box(d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2), float(d.confidence)) for d in detector.predict(image)]


def recognize(reader: Any, plate) -> PlateText:
    # The OCR takes an array to be RGB already: it converts only image files.
    prediction = reader.run_one(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), return_confidence=True, remove_pad_char=False)
    return PlateText(prediction.plate, tuple(float(p) for p in prediction.char_probs))


def main() -> None:
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # What the libraries print must not reach the replies.

    def reply(ok: bool, value: Any) -> None:
        pickle.dump((ok, value), replies, protocol=pickle.HIGHEST_PROTOCOL)
        replies.flush()

    try:
        detector, reader = create_models(**json.loads(sys.argv[1]))
    except Exception as error:
        reply(False, f"{type(error).__name__}: {error}")
        return
    reply(True, detector.model.get_providers())
    methods = {"detect": lambda image: detect(detector, image), "recognize": lambda plate: recognize(reader, plate)}
    while True:
        try:
            method, arguments = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            reply(True, methods[method](*arguments))
        except Exception as error:
            reply(False, f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
