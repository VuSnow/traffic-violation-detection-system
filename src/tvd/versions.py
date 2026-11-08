"""What produced a result: the code, the libraries that shape it, and the model files. Records and the health check keep them."""

from __future__ import annotations

import os
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tvd.config import ComponentConfig
    from tvd.detection import BaseDetector
    from tvd.plates import BasePlateModels

PACKAGES = (
    "traffic-violation-detection", "torch", "ultralytics", "boxmot", "onnxruntime-gpu", "fast-plate-ocr", "open-image-models", "opencv-python", "numpy",
)


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """The installed version of each package, or None if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_version() -> str | None:
    """The commit the code was built from, which the Docker image keeps in ``TVD_COMMIT``. None outside an image."""
    return os.environ.get("TVD_COMMIT") or None


def models_used(detector: BaseDetector, tracker: ComponentConfig, plate_models: BasePlateModels | None, plates_unavailable: str) -> dict[str, Any]:
    """Every model and library behind a record: changing any of them may change what is found."""
    return {
        "detector": detector.describe(),
        "tracker": {"type": tracker.type, **tracker.options},
        "plates": plate_models.describe() if plate_models is not None else {"unavailable": plates_unavailable},
        "software": package_versions(),
        "code": code_version(),
    }
