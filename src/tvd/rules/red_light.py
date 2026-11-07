"""Red-light rule, developed in Notebook 05b. Its output is a candidate for review, never a verdict.

A crossing of the stop line is a candidate only if all of these hold:

- the signal of the vehicle's lane is readable and red when the vehicle enters;
- the vehicle travels on past the line (one that stops on the line is not flagged);
- it does not turn right (a right turn on red may be legal);
- its box does not touch the image border, and it crosses the part of the line that can be judged.

When information is missing, the crossing is not evaluable, never a violation. Version 2 of the rule
(``join_fragments`` and ``estimate_hidden_entry``) also handles a vehicle whose track breaks.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator
from scipy.ndimage import median_filter

from tvd.config import StrictModel, validate
from tvd.geometry import polyline_y
from tvd.motion import CameraMotion
from tvd.observers import FrameObserver
from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult
from tvd.signals import SIGNAL_READERS, UNKNOWN, SignalMonitor, SignalTimeline
from tvd.tracking.fragments import join_fragments
from tvd.tracking.history import TrackPoint, majority_class

SIGNAL_GROUPS = ("through", "left")

CANDIDATE = "candidate"
COMPLIANT = "signal_not_red"
STOPPED = "stopped_at_line"
RIGHT_TURN = "right_turn_on_red"
UNKNOWN_SIGNAL = "not_evaluable_signal"
AT_BORDER = "not_evaluable_border"
OUT_OF_RANGE = "not_evaluable_occluded_line"
ENDED = "not_evaluable_ended"


class SignalHeads(StrictModel):
    """The signal heads of each group, read by one reader. ``options`` go to every head's reader, for example ``half``."""

    reader: str
    options: dict[str, Any] = Field(default_factory=dict)
    through: list[dict[str, Any]] = Field(min_length=1)
    left: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("options", mode="before")
    @classmethod
    def no_options_when_left_empty(cls, value: Any) -> Any:
        return {} if value is None else value


class RedLightScene(StrictModel):
    """Where the stop line, the lanes and the signals are. With a moving camera, the line is drawn on the reference frame."""

    stop_line: list[tuple[float, float]] = Field(min_length=2)
    lane_dividers: list[float] = Field(min_length=1)  # Left of the first: the left-turn lane. Right of the last: the right-turn lane.
    lane_direction_deg: float                         # Direction of travel in the image, in degrees, with y pointing down.
    traffic_direction: Literal["away_from_camera", "toward_camera"]
    evaluable_x_max: float | None = None              # Crossings right of this cannot be judged, for example behind a pole.
    signal_source: str = "camera"
    signals: SignalHeads

    @field_validator("stop_line", "lane_dividers")
    @classmethod
    def increasing_x(cls, value: list) -> list:
        xs = [v[0] if isinstance(v, tuple) else v for v in value]
        if xs != sorted(xs):
            raise ValueError("x must increase from left to right")
        return value


class RedLightSettings(StrictModel):
    version: int
    entry_offset: float = Field(ge=0)          # The front wheels: this fraction of the box height above its bottom.
    pass_margin: float = Field(ge=0)           # The rear must pass the line by this many box heights.
    min_track_frames: int = Field(ge=2)
    smoothing_frames: int = Field(ge=1)
    min_frames_to_decide: int = Field(ge=1)    # Fewer frames left than this: stopped and still moving cannot be told apart.
    still_box_heights: float = Field(ge=0)     # Stopped: moved less than this many box heights over those frames.
    min_heading_px: float = Field(ge=0)        # A shorter movement after the line: heading unknown.
    turn_angle_deg: float = Field(gt=0, lt=180)
    border_margin: float = Field(ge=0)
    duplicate_seconds: float = Field(ge=0)
    duplicate_px: float = Field(ge=0)
    join_fragments: bool
    join_max_gap_frames: int = Field(ge=0)
    join_max_jump_heights: float = Field(ge=0)
    join_max_size_ratio: float = Field(ge=1)
    estimate_hidden_entry: bool
    hidden_fit_frames: int = Field(ge=5)
    hidden_max_frames: int = Field(ge=0)
    hidden_margin_frames: int = Field(ge=0)
    hidden_min_speed: float = Field(ge=0)
    signal_smoothing_frames: int = Field(ge=1)


def movement_after(displacement, lane_direction_deg: float, min_heading_px: float, turn_angle_deg: float) -> str:
    """Movement relative to the lane direction: image axes mislead, because lanes run diagonally in the image."""
    dx, dy = displacement
    if np.hypot(dx, dy) < min_heading_px:
        return "unknown"
    heading = np.degrees(np.arctan2(dy, dx))
    turn = (heading - lane_direction_deg + 180) % 360 - 180
    if turn > turn_angle_deg:
        return "right"
    return "left" if turn < -turn_angle_deg else "straight"


def hidden_entry(frames, bottom, front_margin, box_heights, lane_direction_deg: float, s: RedLightSettings) -> tuple[int, float] | None:
    """The frame and column where a vehicle crossed the line, for a track that starts past it.

    The track's first frames are followed back to the line. None means no estimate: the vehicle is
    waiting rather than leaving the line, it is not moving along the lane, or the line is too far back.
    """
    count = min(s.hidden_fit_frames, len(frames))
    if count < 5:
        return None
    speed = float(np.polyfit(frames[:count], front_margin[:count], 1)[0])  # Negative: moving further past the line.
    if speed >= -s.hidden_min_speed * box_heights[0]:
        return None
    frames_back = float(front_margin[0]) / speed
    if frames_back > s.hidden_max_frames:
        return None
    if movement_after(bottom[count - 1] - bottom[0], lane_direction_deg, 0, s.turn_angle_deg) != "straight":
        return None
    sideways = float(np.polyfit(frames[:count], bottom[:count, 0], 1)[0])
    return int(round(frames[0] - frames_back)), float(bottom[0, 0] - sideways * frames_back)


def reference_box(point: TrackPoint, motion: CameraMotion) -> list[float]:
    """A box moved into reference pixels. It keeps its size, and its bottom centre decides where it lands."""
    x1, y1, x2, y2 = point.box.as_list()
    x, y = motion.to_reference(point.frame_index, [[(x1 + x2) / 2, y2]])[0]
    width, height = x2 - x1, y2 - y1
    return [float(x - width / 2), float(y - height), float(x + width / 2), float(y)]


@RULES.register("red_light")
class RedLightRule(BaseRule):
    name = "red_light"

    def __init__(self, scene: dict[str, Any], settings: dict[str, Any]) -> None:
        self.scene = validate(RedLightScene, scene, "scene")
        self.settings = validate(RedLightSettings, settings, "settings")
        heads = self.scene.signals
        readers = {group: [SIGNAL_READERS.create(heads.reader, **heads.options, **head) for head in getattr(heads, group)] for group in SIGNAL_GROUPS}
        self._monitor = SignalMonitor(readers, self.settings.signal_smoothing_frames)

    def frame_observers(self) -> list[FrameObserver]:
        return [self._monitor]

    def evaluate(self, observation: Observation) -> RuleResult:
        return self.judge(observation, self._monitor.timeline())

    def judge(self, observation: Observation, signals: SignalTimeline) -> RuleResult:
        """Judge every track against the stop line, with signal states read elsewhere."""
        s, motion = self.settings, observation.motion
        tracks = {str(t.track_id): t.points for t in observation.tracks.values()}
        if s.join_fragments:
            box_of = None if motion.is_fixed else (lambda point: reference_box(point, motion))
            tracks = join_fragments(tracks, s.join_max_gap_frames, s.join_max_jump_heights, s.join_max_size_ratio, box_of)
        records = [self.crossing(track_id, points, observation, signals) for track_id, points in tracks.items()]
        records = self.deduplicate([r for r in records if r], observation.fps)
        findings = [self._finding(r, tracks[r["track_id"]], observation.fps) for r in records if r["verdict"] == CANDIDATE]
        return RuleResult(findings, records, s.model_dump())

    def lane_of(self, x: float) -> str:
        dividers = self.scene.lane_dividers
        if x < dividers[0]:
            return "left_turn"
        return "right_turn" if x >= dividers[-1] else "through"

    def crossing(self, track_id: str, points: list[TrackPoint], observation: Observation, signals: SignalTimeline) -> dict | None:
        """How one track crossed the stop line, with the rule's verdict, or None if it never crosses.

        Coming toward the camera, a box's bottom edge is already the vehicle's front, so such a camera
        sets ``entry_offset`` to 0. An estimated entry is judged only if one signal state lasted from
        before it until the vehicle was seen.
        """
        s, scene = self.settings, self.scene
        if len(points) < s.min_track_frames:
            return None
        frames = np.array([p.frame_index for p in points])
        frame_boxes = np.array([p.box.as_list() for p in points], dtype=np.float64)
        boxes = frame_boxes if observation.motion.is_fixed else np.array([reference_box(p, observation.motion) for p in points], dtype=np.float64)
        centers = (boxes[:, 0] + boxes[:, 2]) / 2
        kernel = s.smoothing_frames
        bottom = median_filter(np.c_[centers, boxes[:, 3]], size=(kernel, 1), mode="nearest")
        box_heights = median_filter(boxes[:, 3] - boxes[:, 1], size=kernel, mode="nearest")
        line = polyline_y(scene.stop_line, bottom[:, 0])
        side = -1.0 if scene.traffic_direction == "toward_camera" else 1.0  # Which side of the line is "before" it.
        rear_margin = side * (bottom[:, 1] - line)                         # Positive: the rear is still before the line.
        front_margin = rear_margin - s.entry_offset * box_heights          # Positive: the front is still before the line.

        front_cross = np.flatnonzero((front_margin[:-1] > 0) & (front_margin[1:] <= 0))
        rear_cross = np.flatnonzero((rear_margin[:-1] > 0) & (rear_margin[1:] <= 0))
        estimated = None
        if len(front_cross):
            index, entry_seen = int(front_cross[0]), True
        elif front_margin[0] <= 0 and len(rear_cross):
            # The track starts with the front already past the line: a fragment of an earlier track of this vehicle.
            index, entry_seen = int(rear_cross[0]), False
        elif s.estimate_hidden_entry and front_margin[0] <= 0:
            estimated = hidden_entry(frames, bottom, front_margin, box_heights, scene.lane_direction_deg, s)
            if estimated is None or estimated[0] - s.hidden_margin_frames < observation.first_frame:
                return None
            index, entry_seen = -1, False
        else:
            return None

        at = index + 1
        frame, x = estimated if estimated else (int(frames[at]), float(bottom[at, 0]))
        stop_xs = [p[0] for p in scene.stop_line]
        if not (stop_xs[0] - 30 <= x <= stop_xs[-1] + 30):
            return None

        lane = self.lane_of(x)
        group = "left" if lane == "left_turn" else "through"
        state = signals.state(group, frame)
        if estimated:
            seen = {signals.state(group, k) for k in range(frame - s.hidden_margin_frames, int(frames[0]) + 1)}
            state = seen.pop() if len(seen) == 1 else "conflict"
        state_since = frame
        while signals.get(group, state_since - 1) == state:
            state_since -= 1
        movement = movement_after(bottom[-1] - bottom[at], scene.lane_direction_deg, s.min_heading_px, s.turn_angle_deg)
        box = frame_boxes[at]
        margin = s.border_margin
        at_border = bool(box[0] <= margin or box[2] >= observation.width - margin or box[3] >= observation.height - margin)
        later = rear_margin[at:]
        passed = bool(later.min() <= -s.pass_margin * box_heights[at])
        decide = s.min_frames_to_decide
        recent_motion = float(np.hypot(*(bottom[-1] - bottom[-min(decide, len(later))])))
        still = recent_motion <= s.still_box_heights * box_heights[at]
        evaluable_x_max = observation.width if scene.evaluable_x_max is None else scene.evaluable_x_max

        if at_border:
            verdict = AT_BORDER
        elif x > evaluable_x_max:
            verdict = OUT_OF_RANGE
        elif state in (UNKNOWN, "conflict"):
            verdict = UNKNOWN_SIGNAL
        elif state != "red":
            verdict = COMPLIANT
        elif movement == "right":
            verdict = RIGHT_TURN
        elif passed:
            verdict = CANDIDATE
        elif len(later) < decide or observation.end_frame - frame < decide or not still:
            verdict = ENDED  # A vehicle still moving when its track ends may yet pass the line: that is not a stop.
        else:
            verdict = STOPPED
        return {
            "track_id": track_id, "class_name": majority_class(points), "frame": frame, "x": round(x, 1),
            "lane": lane, "signal_group": group, "signal_state": state, "state_for_frames": frame - state_since, "movement": movement,
            "passed_line": passed, "entry_seen": entry_seen, "entry_estimated": estimated is not None, "verdict": verdict,
            "box_xyxy": [round(float(v), 1) for v in box],
        }

    def deduplicate(self, records: list[dict], fps: float) -> list[dict]:
        """Merge candidates close in time and space: tracker fragments of one vehicle."""
        s = self.settings
        kept = []
        for record in sorted(records, key=lambda r: r["frame"]):
            if record["verdict"] == CANDIDATE and any(
                k["verdict"] == CANDIDATE and (record["frame"] - k["frame"]) / fps <= s.duplicate_seconds and abs(record["x"] - k["x"]) <= s.duplicate_px
                for k in kept
            ):
                continue
            kept.append(record)
        return kept

    def _finding(self, record: dict, points: list[TrackPoint], fps: float) -> Finding:
        entered = "Estimated to have entered, while hidden from view," if record["entry_estimated"] else "Entered"
        return Finding(
            rule=self.name,
            track_id=record["track_id"],
            class_name=record["class_name"],
            start_frame=points[0].frame_index,
            trigger_frame=record["frame"],
            end_frame=points[-1].frame_index,
            why=(f"{entered} in lane '{record['lane']}' while the {record['signal_group']} signal was red "
                 f"({record['state_for_frames'] / fps:.1f} s after it was first seen in that state), and continued past the stop line."),
            evidence={
                **{k: record[k] for k in ("x", "lane", "signal_group", "signal_state", "movement", "entry_seen", "entry_estimated", "box_xyxy")},
                "signal_source": self.scene.signal_source,
            },
        )
