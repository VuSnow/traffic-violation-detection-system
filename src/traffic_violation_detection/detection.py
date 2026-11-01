"""Object detection on a whole video, with the detector settings and drawing style chosen in Notebook 02.

Small functions come first, one for each step. ``annotate_video`` at the end joins them into one
call that goes from a video to a copy with its bounding boxes drawn. That copy is for viewing: later
steps should read the original video, because drawn boxes would cover what they need to see, such
as a license plate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"

DETECTOR_CONFIG = {
    "weights": "yolo26m.pt",
    "imgsz": 1280,
    "conf": 0.10,  # Low on purpose: trackers use low-score boxes. Drawing uses DISPLAY_CONF.
    "iou": 0.7,
    "classes": [0, 1, 2, 3, 5, 7, 9],
}
DISPLAY_CONF = 0.25  # Boxes below this are not drawn.
OUTPUT_CRF = 20      # H.264 quality of the output video: lower is better and larger.

CLASS_COLORS_BGR = {
    "person": (60, 180, 255), "bicycle": (255, 200, 0), "car": (80, 220, 80), "motorcycle": (255, 80, 200),
    "bus": (0, 140, 255), "truck": (40, 40, 220), "traffic light": (0, 255, 255),
}


def load_detector(weights: str = DETECTOR_CONFIG["weights"]):
    """Load a YOLO detector from ``weights/``; Ultralytics downloads known weights on first use."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Detection needs Ultralytics: pip install -e ".[detection]"') from error
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return YOLO(str(WEIGHTS_DIR / weights))


def detect(model, image: np.ndarray, config: dict = DETECTOR_CONFIG) -> list[dict]:
    """Detections of one BGR frame as plain dictionaries. The device is chosen by Ultralytics: GPU if available."""
    result = model.predict(image, imgsz=config["imgsz"], conf=config["conf"], iou=config["iou"], classes=config["classes"], verbose=False)[0]
    boxes = result.boxes
    return [
        {
            "class_id": int(class_id),
            "class_name": model.names[int(class_id)],
            "confidence": round(float(confidence), 4),
            "bbox_xyxy": [round(float(v), 2) for v in xyxy],
        }
        for xyxy, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
    ]


def draw_detections(image: np.ndarray, detections: list[dict], min_conf: float = DISPLAY_CONF) -> None:
    """Draw boxes and labels on a BGR frame, in place. At 1080p this matches Notebook 02; lines and text scale with the height."""
    height = image.shape[0]
    thickness = max(1, round(height / 360))
    font_scale = max(0.5, 0.5 * height / 1080)
    font_thickness = max(1, round(font_scale * 2))
    for d in detections:
        if d["confidence"] < min_conf:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox_xyxy"])
        color = CLASS_COLORS_BGR.get(d["class_name"], (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, (x1, max(0, y1 - text_height - 4)), (x1 + text_width + 2, y1), color, -1)
        cv2.putText(image, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


def annotate_video(video_path: str | Path, output_folder_path: str | Path, model=None) -> Path:
    """Detect objects in every frame of a video, and save a copy with the bounding boxes drawn.

    The output is ``<output_folder_path>/<video name>_output.mp4``, with the size and frame rate
    of the input. It replaces an earlier file of that name, so use one folder per step. The folder
    is created if needed. Returns the path of the output video.

    ``model`` is an already loaded detector, to avoid loading it again for each video.
    """
    video_path = Path(video_path)
    fps = read_video_metadata(video_path)["fps"]  # Fails on a bad path before the slow step of loading the detector.
    detector = model if model is not None else load_detector()
    output_path = output_video_path(video_path, output_folder_path)
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for record in iter_frames(video_path):
            draw_detections(record["image"], detect(detector, record["image"]))
            write_frame(record["image"])
    return output_path
