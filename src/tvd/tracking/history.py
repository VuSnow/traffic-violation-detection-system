"""Tracks as they build up over a video."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from tvd.domain import Box, TrackedObject


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    box: Box
    confidence: float


@dataclass
class Track:
    """Every frame in which one track id was reported."""

    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def class_name(self) -> str:
        """The class reported most often; on a tie, the one reported first."""
        return self.class_counts.most_common(1)[0][0]

    @property
    def first_frame(self) -> int:
        return self.points[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame_index

    def box_at(self, frame_index: int) -> Box | None:
        return next((p.box for p in self.points if p.frame_index == frame_index), None)


class TrackHistory:
    """Collects what a tracker reports, frame by frame, into one ``Track`` per track id."""

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self.frames_seen = 0

    def add(self, frame_index: int, objects: list[TrackedObject]) -> None:
        self.frames_seen += 1
        for tracked in objects:
            track = self._tracks.setdefault(tracked.track_id, Track(tracked.track_id))
            track.points.append(TrackPoint(frame_index, tracked.box, tracked.confidence))
            track.class_counts[tracked.class_name] += 1

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)
