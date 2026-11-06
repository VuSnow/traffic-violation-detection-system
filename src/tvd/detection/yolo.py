"""Ultralytics YOLO models, behind ``BaseDetector``."""

from pathlib import Path

from tvd.detection.base import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame


@DETECTORS.register("yolo")
class YoloDetector(BaseDetector):
    """A YOLO model from Ultralytics.

    Boxes are rounded to 0.01 px and confidences to 4 decimals, as in the notebooks, so that results can be
    compared exactly with theirs.
    """

    def __init__(
        self,
        weights: str,
        image_size: int = 1280,
        confidence: float = 0.10,
        iou: float = 0.7,
        classes: list[str] | None = None,
        weights_dir: str = "weights",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError('The YOLO detector needs Ultralytics: pip install -e ".[detection]"') from error
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(Path(weights_dir) / weights))
        self._image_size = image_size
        self._confidence = confidence
        self._iou = iou
        self._class_ids = None if classes is None else self._ids_of(classes, weights)

    def _ids_of(self, classes: list[str], weights: str) -> list[int]:
        ids = {name: class_id for class_id, name in self._model.names.items()}
        unknown = [name for name in classes if name not in ids]
        if unknown:
            raise ValueError(f"{weights} has no classes named {unknown}")
        return sorted(ids[name] for name in classes)

    def detect(self, frame: Frame) -> list[Detection]:
        result = self._model.predict(
            frame.image, imgsz=self._image_size, conf=self._confidence, iou=self._iou, classes=self._class_ids, verbose=False
        )[0]
        boxes = result.boxes
        return [
            Detection(Box(*(round(float(v), 2) for v in xyxy)), int(class_id), self._model.names[int(class_id)], round(float(confidence), 4))
            for xyxy, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]
