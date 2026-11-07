"""Wrong-way rule, developed in Notebook 05. Its output is a candidate for review, never a verdict.

A track gives a candidate if, in some window (2 s by default), all of these hold:

- C1: the window's midpoint lies in a region with an allowed direction;
- C2: its net movement points against that direction, within the region's tolerance;
- C3: the opposing distance is at least ``min_opposing_heights`` box heights;
- C4: the top edge of the box also moves against the direction, so a box that only grows does not count;
- C5: enough of the window is observed, and no box touches the image border.

Positions are measured in reference pixels, where the regions are drawn, so a moving camera is allowed for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy.ndimage import median_filter

from tvd.config import StrictModel, validate
from tvd.geometry import point_in_polygon
from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult
from tvd.tracking.history import Track

CANDIDATE = "candidate"
NO_VIOLATION = "no_violation"
NOT_EVALUABLE = "not_evaluable"
NOT_IN_REGION = "not_in_directional_region"


class Region(StrictModel):
    """A part of the road where traffic has one allowed direction, drawn on the reference frame."""

    id: str
    polygon: list[tuple[float, float]] = Field(min_length=3)
    allowed_direction: Literal["away_from_camera", "toward_camera"]
    tolerance_deg: float = Field(gt=0, le=180)


class WrongWayScene(StrictModel):
    """Lanes converge in the image, so a direction is given relative to the vanishing point: away from the camera points at it."""

    vanishing_point: tuple[float, float]
    regions: list[Region] = Field(min_length=1)


class WrongWaySettings(StrictModel):
    version: int
    window_seconds: float = Field(gt=0)
    smoothing_seconds: float = Field(ge=0)
    merge_gap_seconds: float = Field(ge=0)
    min_opposing_heights: float = Field(gt=0)
    min_top_edge_share: float = Field(ge=0, le=1)
    min_observed_fraction: float = Field(gt=0, le=1)
    border_margin: int = Field(ge=0)


@dataclass(frozen=True)
class Durations:
    """The rule's durations in frames, for one video's frame rate."""

    window: int
    smoothing: int
    merge_gap: int

    @classmethod
    def at(cls, settings: WrongWaySettings, fps: float) -> Durations:
        # A median filter needs an odd size.
        return cls(round(settings.window_seconds * fps), round(settings.smoothing_seconds * fps) | 1, round(settings.merge_gap_seconds * fps))


@dataclass(frozen=True)
class TrackPath:
    """One vehicle's measurements: its bottom and top centres in reference pixels, its height, and border contact."""

    track_id: str
    class_name: str
    frames: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    heights: np.ndarray
    clipped: np.ndarray


def allowed_direction(region: Region, position: np.ndarray, vanishing_point: tuple[float, float]) -> np.ndarray:
    """Unit vector of the direction a vehicle may travel at a position of a region."""
    position, vanishing = np.asarray(position, dtype=np.float64), np.asarray(vanishing_point, dtype=np.float64)
    unit = (vanishing - position) / np.linalg.norm(vanishing - position)
    return unit if region.allowed_direction == "away_from_camera" else -unit


@RULES.register("wrong_way")
class WrongWayRule(BaseRule):
    name = "wrong_way"

    def __init__(self, scene: dict[str, Any], settings: dict[str, Any]) -> None:
        self.scene = validate(WrongWayScene, scene, "scene")
        self.settings = validate(WrongWaySettings, settings, "settings")

    def evaluate(self, observation: Observation) -> RuleResult:
        durations = Durations.at(self.settings, observation.fps)
        findings, records = [], []
        for track in observation.tracks.values():
            path = self.measure(track, observation)
            verdict, evaluable, groups = self._judge(path, durations)
            records.append({"track_id": path.track_id, "verdict": verdict, "evaluable_windows": evaluable})
            findings += [self._finding(path, group) for group in groups]
        findings.sort(key=lambda f: f.trigger_frame)
        return RuleResult(findings, records, self.settings.model_dump())

    def measure(self, track: Track, observation: Observation) -> TrackPath:
        boxes = np.array([p.box.as_list() for p in track.points], dtype=np.float64)
        frames = np.array([p.frame_index for p in track.points])
        centers = (boxes[:, 0] + boxes[:, 2]) / 2
        motion, m = observation.motion, self.settings.border_margin
        return TrackPath(
            track_id=str(track.track_id),
            class_name=track.class_name,
            frames=frames,
            bottom=np.vstack([motion.to_reference(f, [[x, y]]) for f, x, y in zip(frames, centers, boxes[:, 3])]),
            top=np.vstack([motion.to_reference(f, [[x, y]]) for f, x, y in zip(frames, centers, boxes[:, 1])]),
            heights=boxes[:, 3] - boxes[:, 1],
            clipped=(boxes[:, 0] <= m) | (boxes[:, 1] <= m) | (boxes[:, 2] >= observation.width - m) | (boxes[:, 3] >= observation.height - m),
        )

    def windows(self, path: TrackPath, durations: Durations) -> tuple[list[dict], bool]:
        """Every window whose midpoint is in a region, with its measurements, and whether the track was ever seen in one. No thresholds apply here."""
        frames = path.frames
        kernel = (durations.smoothing, 1)
        bottom = median_filter(path.bottom, size=kernel, mode="nearest")
        top = median_filter(path.top, size=kernel, mode="nearest")
        regions = self.scene.regions
        seen_in_region = any(point_in_polygon(r.polygon, p) for r in regions for p in bottom[::5])
        windows = []
        for i, start_frame in enumerate(frames):
            j = int(np.searchsorted(frames, start_frame + durations.window))
            if j >= len(frames) or frames[j] != start_frame + durations.window:
                continue
            midpoint = (bottom[i] + bottom[j]) / 2
            region = next((r for r in regions if point_in_polygon(r.polygon, midpoint)), None)
            if region is None:
                continue
            allowed = allowed_direction(region, midpoint, self.scene.vanishing_point)
            displacement = bottom[j] - bottom[i]
            distance = float(np.linalg.norm(displacement))
            along = float(-(displacement @ allowed))
            box_height = float(np.median(path.heights[i:j + 1]))
            windows.append({
                "start_frame": int(frames[i]),
                "end_frame": int(frames[j]),
                "region_id": region.id,
                "opposing_px": along,
                "opposing_heights": along / box_height,
                "top_opposing_px": float(-((top[j] - top[i]) @ allowed)),
                "angle_to_opposite_deg": float(np.degrees(np.arccos(np.clip(along / distance, -1, 1)))) if distance > 0 else 180.0,
                "tolerance_deg": region.tolerance_deg,
                "observed_fraction": (j - i + 1) / (durations.window + 1),
                "clipped": bool(path.clipped[i:j + 1].any()),
            })
        return windows, seen_in_region

    def usable(self, window: dict) -> bool:
        """C5: the window can be trusted."""
        return window["observed_fraction"] >= self.settings.min_observed_fraction and not window["clipped"]

    def failed_conditions(self, window: dict) -> list[str]:
        """Which conditions a window fails; an empty list means it passes."""
        s = self.settings
        failed = []
        if window["angle_to_opposite_deg"] > window["tolerance_deg"]:
            failed.append("C2 direction")
        if window["opposing_heights"] < s.min_opposing_heights:
            failed.append("C3 distance")
        if window["opposing_px"] <= 0 or window["top_opposing_px"] < s.min_top_edge_share * window["opposing_px"]:
            failed.append("C4 top edge")
        if not self.usable(window):
            failed.append("C5 quality")
        return failed

    def _judge(self, path: TrackPath, durations: Durations) -> tuple[str, int, list[list[dict]]]:
        """The track's verdict, its number of usable windows, and its passing windows grouped into events.

        ``not_evaluable`` is kept apart from ``no_violation``: "we could not check" differs from "we found nothing".
        """
        windows, seen_in_region = self.windows(path, durations)
        evaluable = [w for w in windows if self.usable(w)]
        passing = [w for w in evaluable if not self.failed_conditions(w)]
        if passing:
            verdict = CANDIDATE
        elif evaluable:
            verdict = NO_VIOLATION
        elif seen_in_region or windows:
            verdict = NOT_EVALUABLE
        else:
            verdict = NOT_IN_REGION
        return verdict, len(evaluable), self._merge(passing, durations)

    def _merge(self, passing: list[dict], durations: Durations) -> list[list[dict]]:
        """Overlapping passing windows of one region form one event."""
        groups: list[list[dict]] = []
        for window in sorted(passing, key=lambda w: w["start_frame"]):
            if groups and window["region_id"] == groups[-1][-1]["region_id"] and window["start_frame"] <= groups[-1][-1]["end_frame"] + durations.merge_gap:
                groups[-1].append(window)
            else:
                groups.append([window])
        return groups

    def _finding(self, path: TrackPath, group: list[dict]) -> Finding:
        first, last = group[0], group[-1]
        best = max(group, key=lambda w: w["opposing_heights"])
        start, trigger, end = first["start_frame"], first["end_frame"], last["end_frame"]
        return Finding(
            rule=self.name,
            track_id=path.track_id,
            class_name=path.class_name,
            start_frame=start,
            trigger_frame=trigger,
            end_frame=end,
            why=(f"Moved against the allowed direction of {first['region_id']} by {best['opposing_heights']:.2f} box heights "
                 f"in about {self.settings.window_seconds:.1f} s, {best['angle_to_opposite_deg']:.0f} degrees from the opposite direction."),
            evidence={
                "region_id": first["region_id"],
                "opposing_heights": round(best["opposing_heights"], 3),
                "opposing_px": round(best["opposing_px"], 1),
                "top_opposing_px": round(best["top_opposing_px"], 1),
                "angle_to_opposite_deg": round(best["angle_to_opposite_deg"], 1),
                "evidence_frames": [start, trigger, end],
            },
        )
