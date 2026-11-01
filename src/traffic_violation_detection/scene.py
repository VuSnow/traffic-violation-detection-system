"""Scene configuration and camera-motion helpers, validated in notebooks/04_traffic_scene_configuration.ipynb.

Conventions:

- Scene geometry is expressed in reference coordinates: pixels of one chosen
  reference frame, with the origin at the top-left and y pointing down.
- A frame's camera motion is a similarity transform ``(a, b, tx, ty)`` mapping
  its pixels to reference pixels: ``x' = a*x - b*y + tx``, ``y' = b*x + a*y + ty``.
- Allowed directions are relative to the scene's vanishing point, because lanes
  converge in the image: ``away_from_camera`` points toward it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

VALID_DIRECTIONS = {"away_from_camera", "toward_camera", None}


def load_scene_config(path: str | Path, width: int, height: int) -> dict:
    """Load a scene configuration and check it against the video's size.

    Raises ValueError for a configuration that does not fit, so a wrong or stale
    file fails loudly instead of producing wrong violations silently.
    """
    scene = json.loads(Path(path).read_text())
    if scene.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {scene.get('schema_version')}")
    if (scene["video"]["width"], scene["video"]["height"]) != (width, height):
        raise ValueError(f"Scene is for {scene['video']['width']}x{scene['video']['height']}, video is {width}x{height}")
    x1, y1, x2, y2 = scene["coordinates"]["always_visible_xyxy"]

    def check_points(owner: str, points) -> None:
        for x, y in points:
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                raise ValueError(f"{owner}: point ({x}, {y}) is outside the always-visible area")

    region_ids = set()
    for region in scene["regions"]:
        if len(region["polygon"]) < 3:
            raise ValueError(f"{region['id']}: a polygon needs at least 3 points")
        if cv2.contourArea(np.array(region["polygon"], dtype=np.float32)) <= 0:
            raise ValueError(f"{region['id']}: polygon has no area")
        if region["allowed_direction"] not in VALID_DIRECTIONS:
            raise ValueError(f"{region['id']}: unknown direction {region['allowed_direction']}")
        check_points(region["id"], region["polygon"])
        region_ids.add(region["id"])
    for line in scene["lines"]:
        if len(line["points"]) < 2:
            raise ValueError(f"{line['id']}: a line needs at least 2 points")
        check_points(line["id"], line["points"])
    for signal in scene["signals"]:
        rx1, ry1, rx2, ry2 = signal["roi_xyxy"]
        if rx2 <= rx1 or ry2 <= ry1:
            raise ValueError(f"{signal['id']}: empty ROI")
        check_points(signal["id"], [[rx1, ry1], [rx2, ry2]])
        unknown = set(signal["governs_regions"]) - region_ids
        if unknown:
            raise ValueError(f"{signal['id']}: governs unknown regions {sorted(unknown)}")
    return scene


def region_by_id(scene: dict, region_id: str) -> dict:
    return next(r for r in scene["regions"] if r["id"] == region_id)


def point_in_polygon(polygon, point) -> bool:
    """True if the point lies inside the polygon or on its boundary."""
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), (float(point[0]), float(point[1])), False) >= 0


def allowed_direction_unit(region: dict, position, vanishing_point) -> np.ndarray:
    """Unit vector of the allowed travel direction at a position in a region that has one."""
    if region["allowed_direction"] is None:
        raise ValueError(f"{region['id']}: no allowed direction")
    position, vanishing_point = np.asarray(position, dtype=np.float64), np.asarray(vanishing_point, dtype=np.float64)
    unit = (vanishing_point - position) / np.linalg.norm(vanishing_point - position)
    return unit if region["allowed_direction"] == "away_from_camera" else -unit


def similarity_matrix(params) -> np.ndarray:
    """2x3 matrix of a similarity transform given as ``(a, b, tx, ty)``."""
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def frame_to_reference(params, points) -> np.ndarray:
    """Map (x, y) points of a frame to reference coordinates."""
    return cv2.transform(np.asarray(points, dtype=np.float64)[None], similarity_matrix(params))[0]


def reference_to_frame(params, points) -> np.ndarray:
    """Map reference coordinates to the pixel coordinates of a frame."""
    inverse = cv2.invertAffineTransform(similarity_matrix(params))
    return cv2.transform(np.asarray(points, dtype=np.float64)[None], inverse)[0]


def load_camera_motion(path: str | Path) -> np.ndarray:
    """Per-frame ``(a, b, tx, ty)`` array from a camera motion file, indexed by frame."""
    frames = json.loads(Path(path).read_text())["frames"]
    if [f["frame_index"] for f in frames] != list(range(len(frames))):
        raise ValueError(f"{path}: camera motion must have one record per frame, in order")
    return np.array([[f["a"], f["b"], f["tx"], f["ty"]] for f in frames], dtype=np.float64)
