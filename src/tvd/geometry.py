"""Plane geometry in pixels, with the origin at the top-left and y pointing down."""

from collections.abc import Sequence

import cv2
import numpy as np

Point = Sequence[float]


def polyline_y(points: Sequence[Point], x):
    """Height of a polyline, such as a stop line, at image column ``x`` (one value or an array).

    Its points must have increasing x. Beyond its ends the height of the nearest end is used.
    """
    xs, ys = np.array(points, dtype=np.float64).T
    return np.interp(x, xs, ys)


def point_in_polygon(polygon: Sequence[Point], point: Point) -> bool:
    """True if the point lies inside the polygon or on its boundary."""
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), (float(point[0]), float(point[1])), False) >= 0
