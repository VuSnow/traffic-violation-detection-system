"""Joining two tracks that are plainly one vehicle under two ids."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tvd.tracking.history import TrackPoint

BoxOf = Callable[[TrackPoint], list[float]]


def join_fragments(
    tracks: dict[str, list[TrackPoint]],
    max_gap_frames: int,
    max_jump_heights: float,
    max_size_ratio: float,
    box_of: BoxOf | None = None,
) -> dict[str, list[TrackPoint]]:
    """Join tracks where one ends and another begins at the same place and size. A joined id is ``"<first>+<second>"``.

    The limits are strict on purpose: joining two vehicles could give one the other's plate, while a
    vehicle left in two parts only costs a missed case. ``box_of`` gives a point's box, in reference
    pixels when the camera moves. The closest continuation is joined first, until none is left.
    """
    box_of = box_of or (lambda point: point.box.as_list())

    def bottom_and_height(point: TrackPoint) -> tuple[float, float, float]:
        x1, y1, x2, y2 = box_of(point)
        return (x1 + x2) / 2, y2, y2 - y1

    def continuation(first: list[TrackPoint], second: list[TrackPoint]) -> float | None:
        """How far the second track starts from the end of the first, in box heights, or None if it does not continue it."""
        starts_later = second[0].frame_index > first[0].frame_index and second[-1].frame_index > first[-1].frame_index
        if not (starts_later and abs(second[0].frame_index - first[-1].frame_index) <= max_gap_frames):
            return None
        x1, y1, height1 = bottom_and_height(first[-1])
        x2, y2, height2 = bottom_and_height(second[0])
        jump = float(np.hypot(x2 - x1, y2 - y1)) / max(height1, 1)
        return jump if jump <= max_jump_heights and max(height1 / height2, height2 / height1) <= max_size_ratio else None

    joined = {track_id: list(points) for track_id, points in tracks.items()}
    while True:
        pairs = [(continuation(joined[a], joined[b]), a, b) for a in joined for b in joined if a != b]
        pairs = [p for p in pairs if p[0] is not None]
        if not pairs:
            return joined
        _, first, second = min(pairs)
        later = joined.pop(second)
        earlier = joined.pop(first)
        joined[f"{first}+{second}"] = [p for p in earlier if p.frame_index < later[0].frame_index] + later
