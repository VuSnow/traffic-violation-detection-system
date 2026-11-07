"""Camera motion from ORB keypoints matched to a reference frame, the method chosen in Notebook 04.

Keypoints on moving objects follow the traffic, not the camera, so every detected object is masked
out. Each frame is matched directly to the reference frame, which avoids accumulating drift.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import cv2
import numpy as np

from tvd.domain import Detection
from tvd.motion.base import IDENTITY, MOTION_ESTIMATORS, BaseMotionEstimator, CameraMotion
from tvd.video import VideoReader


@MOTION_ESTIMATORS.register("orb")
class OrbMotionEstimator(BaseMotionEstimator):
    """A moving camera. About 0.2 s per 1080p frame on 4 CPU cores, so use it only for a camera that moves.

    ``excluded_areas`` are ``[x1, y1, x2, y2]`` boxes of anything drawn over the image, such as a logo or
    a signal graphic, which does not move with the road. ``static_classes`` are detected classes that
    are part of the scene, so their keypoints stay usable.
    """

    needs_detections = True

    def __init__(
        self,
        reference_frame: int,
        excluded_areas: Sequence[Sequence[int]] = (),
        features: int = 6000,
        ratio_test: float = 0.75,
        ransac_threshold: float = 2.0,
        mask_margin: int = 12,
        static_classes: Sequence[str] = ("traffic light",),
        max_threads: int = 32,
    ) -> None:
        self.reference_frame = reference_frame
        self.excluded_areas = [tuple(int(v) for v in area) for area in excluded_areas]
        self.features = features
        self.ratio_test = ratio_test
        self.ransac_threshold = ransac_threshold
        self.mask_margin = mask_margin
        self.static_classes = set(static_classes)
        self.max_threads = max_threads

    def estimate(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> CameraMotion:
        if not start <= self.reference_frame < stop:
            raise ValueError(f"Reference frame {self.reference_frame} is outside the analysed frames {start} to {stop - 1}")
        # Importing Ultralytics leaves OpenCV on one thread. This step then takes about 4 times longer, with the same result.
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(min(self.max_threads, os.cpu_count() or 1))
        try:
            return CameraMotion(self._register(reader, detections, start, stop), first_frame=start)
        finally:
            cv2.setNumThreads(previous_threads)

    def static_mask(self, shape: tuple[int, int], detections: list[Detection]) -> np.ndarray:
        """255 where keypoints may be used: everywhere but the excluded areas and the moving objects."""
        mask = np.full(shape, 255, np.uint8)
        for x1, y1, x2, y2 in self.excluded_areas:
            mask[y1:y2, x1:x2] = 0
        margin = self.mask_margin
        for d in detections:
            if d.class_name in self.static_classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in d.box.as_list())
            mask[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin] = 0
        return mask

    def _fit(self, points, descriptors, reference_points, reference_descriptors, matcher) -> tuple[float, float, float, float] | None:
        """The transform from a frame to the reference, or None if the frame cannot be matched."""
        if descriptors is None or reference_descriptors is None:
            return None
        pairs = matcher.knnMatch(descriptors, reference_descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio_test * p[1].distance]
        if len(good) < 3:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(
            np.float32([points[m.queryIdx].pt for m in good]),
            np.float32([reference_points[m.trainIdx].pt for m in good]),
            method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=5000,
        )
        return None if matrix is None else (float(matrix[0, 0]), float(matrix[1, 0]), float(matrix[0, 2]), float(matrix[1, 2]))

    def _register(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> np.ndarray:
        orb = cv2.ORB_create(self.features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        reference = next(reader.frames(self.reference_frame, self.reference_frame + 1))
        gray = cv2.cvtColor(reference.image, cv2.COLOR_BGR2GRAY)
        reference_points, reference_descriptors = orb.detectAndCompute(gray, self.static_mask(gray.shape, detections.get(self.reference_frame, [])))

        params = np.zeros((stop - start, 4))
        current = IDENTITY
        unmatched = 0
        for frame in reader.frames(start, stop):
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            points, descriptors = orb.detectAndCompute(gray, self.static_mask(gray.shape, detections.get(frame.index, [])))
            fitted = self._fit(points, descriptors, reference_points, reference_descriptors, matcher)
            unmatched += fitted is None
            current = fitted if fitted is not None else current
            params[frame.index - start] = current
        if unmatched:
            warnings.warn(f"Camera motion: {unmatched} of {len(params)} frames could not be matched and reuse the previous transform.", stacklevel=3)
        return params
