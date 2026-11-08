"""What produced a result: the code, the libraries that shape it, and the model files. Records and the health check keep them."""

from __future__ import annotations

import os
from importlib import metadata

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
