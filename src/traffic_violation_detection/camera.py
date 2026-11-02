"""Camera motion: how each frame maps to a reference frame, with the method chosen in Notebook 04.

Keypoints on moving objects would follow the traffic and not the camera, so detected objects are
masked out. Each frame is matched directly to the reference, which avoids accumulating drift.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np

from traffic_violation_detection.video import iter_frames, read_frame_at

MASK_MARGIN = 12      # px added around each masked box
ORB_FEATURES = 6000
RATIO_TEST = 0.75
RANSAC_THRESHOLD = 2.0
IDENTITY = (1.0, 0.0, 0.0, 0.0)
STATIC_CLASSES = {"traffic light"}  # Part of the scene, so they stay usable.


def static_mask(shape: tuple[int, int], detections: list[dict], margin: int = MASK_MARGIN, excluded_areas=()) -> np.ndarray:
    """A mask of the image without its moving objects: 255 where keypoints may be used.

    ``excluded_areas`` are ``x1, y1, x2, y2`` boxes of anything that does not move with the camera, such as a watermark.
    """
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in excluded_areas:
        mask[y1:y2, x1:x2] = 0
    for d in detections:
        if d["class_name"] in STATIC_CLASSES:
            continue
        x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
        mask[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin] = 0
    return mask


def fit_similarity(points, descriptors, reference_points, reference_descriptors, matcher) -> tuple[float, float, float, float] | None:
    """The transform ``(a, b, tx, ty)`` from a frame to the reference, or None if the frame cannot be matched."""
    if descriptors is None or reference_descriptors is None:
        return None
    pairs = matcher.knnMatch(descriptors, reference_descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < 3:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        np.float32([points[m.queryIdx].pt for m in good]),
        np.float32([reference_points[m.trainIdx].pt for m in good]),
        method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD, maxIters=5000,
    )
    return None if matrix is None else (float(matrix[0, 0]), float(matrix[1, 0]), float(matrix[0, 2]), float(matrix[1, 2]))


def estimate_camera_motion(video_path: str | Path, detections_by_frame: list[list[dict]], reference_frame: int, excluded_areas=()) -> np.ndarray:
    """Per-frame ``(a, b, tx, ty)`` from each frame to the reference frame, for the frames of ``detections_by_frame``.

    A frame that cannot be matched keeps the transform of the frame before it, and a warning reports how many did.
    """
    if not 0 <= reference_frame < len(detections_by_frame):
        raise ValueError(f"Reference frame {reference_frame} is outside the video's {len(detections_by_frame)} frames")
    orb = cv2.ORB_create(ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    gray = cv2.cvtColor(read_frame_at(video_path, reference_frame)["image"], cv2.COLOR_BGR2GRAY)
    reference_points, reference_descriptors = orb.detectAndCompute(gray, static_mask(gray.shape, detections_by_frame[reference_frame], excluded_areas=excluded_areas))

    params = np.zeros((len(detections_by_frame), 4))
    current = IDENTITY
    unmatched = 0
    for record in iter_frames(video_path, 0, len(detections_by_frame)):
        k = record["frame_index"]
        gray = cv2.cvtColor(record["image"], cv2.COLOR_BGR2GRAY)
        points, descriptors = orb.detectAndCompute(gray, static_mask(gray.shape, detections_by_frame[k], excluded_areas=excluded_areas))
        fitted = fit_similarity(points, descriptors, reference_points, reference_descriptors, matcher)
        unmatched += fitted is None
        current = fitted if fitted is not None else current
        params[k] = current
    if unmatched:
        warnings.warn(f"Camera motion: {unmatched} of {len(params)} frames could not be matched and reuse the previous transform.", stacklevel=2)
    return params
