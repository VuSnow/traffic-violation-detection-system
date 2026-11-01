"""Red-light rule: signal reading and stop-line crossing logic, developed in notebooks/05b_red_light_violation.ipynb.

The rule was shaped on three clips by inspecting its mistakes, then frozen before being
evaluated on the other clips. Its output is a candidate for human review, never a verdict.

A crossing is a candidate only if all of these hold:

- the signal of the vehicle's lane is readable and red when the vehicle enters;
- the vehicle travels on past the line (a vehicle that stops on the line is not flagged);
- it does not turn right (a right turn on red may be legal);
- its box does not touch the image border, and the crossing is in the evaluable part of the line.

When information is missing, the crossing is reported as not evaluable, never as a violation.

Small functions come first. ``detect_red_light_video`` at the end joins them into one call that goes
from a video to a copy showing the result, one clip per candidate, and a file of events.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from traffic_violation_detection.detection import OUTPUT_CRF, load_detector
from traffic_violation_detection.evidence import write_event_clips
from traffic_violation_detection.tracking import TRACKING_SESSION, add_to_tracks, majority_class, track_frames
from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer

RULE = {
    "name": "red_light",
    "version": 1,
    "entry_offset": 0.30,      # front wheels: this fraction of the box height above the box bottom
    "pass_margin": 0.5,        # the rear must pass the line by this many box heights
    "min_track_frames": 8,
    "smoothing_frames": 5,
    "min_frames_to_decide": 15,  # fewer frames left than this: cannot tell stopped from still moving
    "still_box_heights": 0.15, # stopped: moved less than this many box heights over those last frames
    "min_heading_px": 60,      # shorter movement after the line: heading unknown
    "turn_angle_deg": 35,
    "border_margin": 3,
    "duplicate_seconds": 1.0,
    "duplicate_px": 120,
}

SIGNAL_GROUPS = ("through", "left")

CANDIDATE = "candidate"
COMPLIANT = "signal_not_red"
STOPPED = "stopped_at_line"
RIGHT_TURN = "right_turn_on_red"
UNKNOWN_SIGNAL = "not_evaluable_signal"
AT_BORDER = "not_evaluable_border"
OUT_OF_RANGE = "not_evaluable_occluded_line"
ENDED = "not_evaluable_ended"


def load_red_light_scene(path: str | Path, width: int, height: int) -> dict:
    """Load a per-clip red-light configuration and check it against the video's size."""
    scene = json.loads(Path(path).read_text())
    if scene.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {scene.get('schema_version')}")
    if (scene["video"]["width"], scene["video"]["height"]) != (width, height):
        raise ValueError(f"Scene is for {scene['video']['width']}x{scene['video']['height']}, video is {width}x{height}")
    for name, clip in scene["clips"].items():
        xs = [p[0] for p in clip["stop_line"]]
        if len(xs) < 2 or xs != sorted(xs):
            raise ValueError(f"clip {name}: the stop line needs at least 2 points with increasing x")
        if clip["start"] >= clip["end"]:
            raise ValueError(f"clip {name}: empty frame range")
        if clip["signals"]["method"] not in ("lamps", "head"):
            raise ValueError(f"clip {name}: unknown signal method {clip['signals']['method']}")
        if not clip["signals"]["through"]:
            raise ValueError(f"clip {name}: no signal for the through lanes")
    return scene


def read_head_state(patch: np.ndarray) -> str:
    """Night: colour of the glow around a signal head. A monochrome (infrared) image gives ``unknown``."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if saturation.mean() < 8:
        return "unknown"
    lit = (value >= 120) & (saturation >= 70)
    counts = {
        "red": int((lit & ((hue <= 10) | (hue >= 165))).sum()),
        "yellow": int((lit & (hue >= 14) & (hue <= 35)).sum()),
        "green": int((lit & (hue >= 40) & (hue <= 105)).sum()),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] >= 12 and counts[best] >= 2 * (sum(counts.values()) - counts[best]) else "unknown"


def read_lamp_state(frame: np.ndarray, lamps: dict, half: int = 6) -> str:
    """Day: each lamp position separately, because a head's yellow backplate confuses a whole-head count."""
    counts = {}
    height, width = frame.shape[:2]
    for colour, (x, y) in lamps.items():
        if not (half <= x < width - half and half <= y < height - half):
            counts[colour] = 0
            continue
        hsv = cv2.cvtColor(frame[y - half:y + half + 1, x - half:x + half + 1], cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if colour == "red":
            counts[colour] = int((((hue <= 10) | (hue >= 165)) & (saturation >= 90) & (value >= 90)).sum())
        elif colour == "yellow":
            counts[colour] = int(((hue >= 14) & (hue <= 35) & (saturation >= 120) & (value >= 170)).sum())
        else:
            counts[colour] = int(((hue >= 40) & (hue <= 105) & (saturation >= 60) & (value >= 110)).sum())
    best = max(counts, key=counts.get)
    return best if counts[best] >= 6 and counts[best] >= 2 * (sum(counts.values()) - counts[best]) else "unknown"


def read_group_state(frame: np.ndarray, method: str, heads: list) -> str:
    """Combined state of the heads of one signal group: ``conflict`` if they disagree, ``unknown`` if none is readable."""
    if method == "lamps":
        states = [read_lamp_state(frame, lamps) for lamps in heads]
    else:
        states = [read_head_state(frame[y1:y2, x1:x2]) for x1, y1, x2, y2 in heads]
    known = {s for s in states if s != "unknown"}
    if not known:
        return "unknown"
    return known.pop() if len(known) == 1 else "conflict"


def smooth_states(states: list[str], window: int = 7) -> list[str]:
    """Majority vote over a sliding window, to remove single-frame flicker."""
    half = window // 2
    smoothed = []
    for i in range(len(states)):
        votes = states[max(0, i - half):i + half + 1]
        smoothed.append(max(set(votes), key=lambda s: (votes.count(s), s)))
    return smoothed


def line_y(stop_line, x):
    """Height of the stop line at image column x."""
    xs, ys = np.array(stop_line, dtype=np.float64).T
    return np.interp(x, xs, ys)


def lane_of(clip: dict, x: float) -> str:
    dividers = clip["lane_dividers"]
    if x < dividers[0]:
        return "left_turn"
    return "right_turn" if x >= dividers[-1] else "through"


def movement_after(displacement, lane_direction_deg: float, rule: dict = RULE) -> str:
    """Movement relative to the lane direction (image axes mislead, as lanes run diagonally in the image)."""
    dx, dy = displacement
    if np.hypot(dx, dy) < rule["min_heading_px"]:
        return "unknown"
    heading = np.degrees(np.arctan2(dy, dx))
    turn = (heading - lane_direction_deg + 180) % 360 - 180
    if turn > rule["turn_angle_deg"]:
        return "right"
    return "left" if turn < -rule["turn_angle_deg"] else "straight"


def evaluate_track(track: dict, clip: dict, signal_states: dict, lane_direction_deg: float, width: int, height: int, rule: dict = RULE) -> dict | None:
    """Judge one track against the stop line. Returns a crossing record, or None if it never crosses.

    ``track`` holds ``track_id``, ``frames``, ``boxes`` (N x 4, xyxy) and ``class_name``.
    ``signal_states`` maps ``"through"`` and ``"left"`` to per-frame states, indexed by frame number.
    """
    frames = np.asarray(track["frames"])
    boxes = np.asarray(track["boxes"], dtype=np.float64)
    if len(frames) < rule["min_track_frames"]:
        return None
    centers = (boxes[:, 0] + boxes[:, 2]) / 2
    kernel = rule["smoothing_frames"]
    bottom = median_filter(np.c_[centers, boxes[:, 3]], size=(kernel, 1), mode="nearest")
    box_heights = median_filter(boxes[:, 3] - boxes[:, 1], size=kernel, mode="nearest")
    line = line_y(clip["stop_line"], bottom[:, 0])
    rear_margin = bottom[:, 1] - line                                       # positive: the rear is still before the line
    front_margin = rear_margin - rule["entry_offset"] * box_heights          # positive: the front is still before the line

    front_cross = np.flatnonzero((front_margin[:-1] > 0) & (front_margin[1:] <= 0))
    rear_cross = np.flatnonzero((rear_margin[:-1] > 0) & (rear_margin[1:] <= 0))
    if len(front_cross):
        index, entry_seen = int(front_cross[0]), True
    elif front_margin[0] <= 0 and len(rear_cross):
        # The track starts with the front already past the line: a fragment of an earlier track of this vehicle.
        index, entry_seen = int(rear_cross[0]), False
    else:
        return None

    at = index + 1
    frame = int(frames[at])
    x = float(bottom[at, 0])
    stop_xs = [p[0] for p in clip["stop_line"]]
    if not (stop_xs[0] - 30 <= x <= stop_xs[-1] + 30):
        return None

    lane = lane_of(clip, x)
    group = "left" if lane == "left_turn" else "through"
    state = signal_states[group].get(frame, "unknown")
    state_since = frame
    while signal_states[group].get(state_since - 1) == state:
        state_since -= 1
    movement = movement_after(bottom[-1] - bottom[at], lane_direction_deg, rule)
    box = boxes[at]
    margin = rule["border_margin"]
    at_border = bool(box[0] <= margin or box[2] >= width - margin or box[3] >= height - margin)
    later = rear_margin[at:]
    passed = bool(later.min() <= -rule["pass_margin"] * box_heights[at])
    decide = rule["min_frames_to_decide"]
    recent_motion = float(np.hypot(*(bottom[-1] - bottom[-min(decide, len(later))])))
    still = recent_motion <= rule["still_box_heights"] * box_heights[at]

    if at_border:
        verdict = AT_BORDER
    elif x > clip.get("evaluable_x_max", width):
        verdict = OUT_OF_RANGE
    elif state in ("unknown", "conflict"):
        verdict = UNKNOWN_SIGNAL
    elif state != "red":
        verdict = COMPLIANT
    elif movement == "right":
        verdict = RIGHT_TURN
    elif passed:
        verdict = CANDIDATE
    elif len(later) < decide or clip["end"] - frame < decide or not still:
        verdict = ENDED  # A vehicle still moving when its track ends may yet pass the line: that is not a stop.
    else:
        verdict = STOPPED
    return {
        "track_id": str(track["track_id"]), "class_name": track.get("class_name", "vehicle"), "frame": frame, "x": round(x, 1),
        "lane": lane, "signal_group": group, "signal_state": state, "state_for_frames": frame - state_since, "movement": movement,
        "passed_line": passed, "entry_seen": entry_seen, "verdict": verdict, "box_xyxy": [round(float(v), 1) for v in box],
    }


def deduplicate(records: list[dict], fps: float, rule: dict = RULE) -> list[dict]:
    """Merge candidates that are close in time and space: tracker fragments of one vehicle."""
    kept = []
    for record in sorted(records, key=lambda r: r["frame"]):
        if record["verdict"] == CANDIDATE and any(
            k["verdict"] == CANDIDATE and (record["frame"] - k["frame"]) / fps <= rule["duplicate_seconds"] and abs(record["x"] - k["x"]) <= rule["duplicate_px"]
            for k in kept
        ):
            continue
        kept.append(record)
    return kept


def score_candidates(candidates: list[dict], events: list[dict], max_frame_difference: int, max_x_difference: float) -> dict:
    """Match candidates to labelled events one-to-one, closest in time first, and sort them into outcomes.

    A candidate matched to no label, or to a ``not_violation`` label, is a false positive. One matched
    to an ``uncertain`` label is ignored. A ``violation`` label with no candidate is a false negative.
    A second candidate for the same vehicle finds its label taken, so it counts as a false positive.
    """
    pairs = sorted(
        (abs(c["frame"] - e["frame"]), abs(c["x"] - e["x"]), i, j)
        for i, c in enumerate(candidates)
        for j, e in enumerate(events)
        if abs(c["frame"] - e["frame"]) <= max_frame_difference and abs(c["x"] - e["x"]) <= max_x_difference
    )
    matched, used = {}, set()
    for _, _, i, j in pairs:
        if i not in matched and j not in used:
            matched[i] = j
            used.add(j)
    outcome = {"true_positives": [], "false_positives": [], "ignored": []}
    for i, candidate in enumerate(candidates):
        event = events[matched[i]] if i in matched else None
        if event is not None and event["label"] == "violation":
            outcome["true_positives"].append((candidate, event))
        elif event is not None and event["label"] == "uncertain":
            outcome["ignored"].append((candidate, event))
        else:
            outcome["false_positives"].append((candidate, event))
    outcome["false_negatives"] = [e for j, e in enumerate(events) if e["label"] == "violation" and j not in used]
    return outcome


def build_event(record: dict, clip_name: str, video_id: str, fps: float, clip_start: int, signal_source: str = "camera", rule: dict = RULE) -> dict:
    """A reviewable candidate event. Its status is always pending, and its id never reuses the track id alone."""
    frame = record["frame"]
    return {
        "event_id": f"evt-{video_id}-{clip_name}-{rule['name']}-f{frame:05d}-t{record['track_id']}",
        "video_id": video_id,
        "clip": clip_name,
        "violation_type": rule["name"],
        "track_id": record["track_id"],
        "class_name": record["class_name"],
        "trigger_frame": frame,
        "trigger_time_s": round(frame / fps, 3),
        "time_in_clip_s": round((frame - clip_start) / fps, 3),
        "rule": {"name": rule["name"], "version": rule["version"]},
        "signal_source": signal_source,
        "why": (f"Entered in lane '{record['lane']}' while the {record['signal_group']} signal was red "
                f"({record['state_for_frames'] / fps:.1f} s after it was first seen in that state), and continued past the stop line."),
        "evidence": {k: record[k] for k in ("x", "lane", "signal_group", "signal_state", "movement", "entry_seen", "box_xyxy")},
        "license_plate": None,
        "review_status": "pending",
    }


def read_frame_signals(image: np.ndarray, clip: dict) -> dict[str, str]:
    """The state of each signal group in one frame."""
    signals = clip["signals"]
    return {group: read_group_state(image, signals["method"], signals[group]) if signals[group] else "unknown" for group in SIGNAL_GROUPS}


def track_and_read_clip(video_path: str | Path, clip: dict, detector) -> tuple[dict[str, list], dict[str, dict[int, str]]]:
    """One pass over a clip, with a new tracker. Returns its tracks, and each signal group's smoothed state per frame."""
    tracks: dict[str, list] = {}
    raw: dict[str, list[str]] = {group: [] for group in SIGNAL_GROUPS}
    for record, _, tracked in track_frames(video_path, detector, clip["start"], clip["end"]):
        add_to_tracks(tracks, record["frame_index"], tracked)
        for group, state in read_frame_signals(record["image"], clip).items():
            raw[group].append(state)
    states = {group: {clip["start"] + i: s for i, s in enumerate(smooth_states(values))} for group, values in raw.items()}
    return tracks, states


def as_track(track_id: str, rows: list) -> dict:
    """The input of ``evaluate_track`` from a track's rows."""
    return {"track_id": track_id, "class_name": majority_class(rows), "frames": [r[0] for r in rows], "boxes": [r[1:5] for r in rows]}


def evaluate_clip(tracks: dict[str, list], clip: dict, states: dict, lane_direction_deg: float, width: int, height: int, fps: float, rule: dict = RULE) -> list[dict]:
    """The crossing records of a clip's tracks, with duplicate candidates merged."""
    records = [evaluate_track(as_track(t, rows), clip, states, lane_direction_deg, width, height, rule) for t, rows in tracks.items()]
    return deduplicate([r for r in records if r], fps, rule)


def verdict_color(verdict: str) -> tuple[int, int, int]:
    if verdict == CANDIDATE:
        return (40, 40, 230)
    return (170, 170, 170) if verdict.startswith("not_evaluable") else (80, 200, 80)


def draw_clip_frame(image: np.ndarray, clip: dict, present: list[tuple[str, list]], records_by_track: dict[str, dict], frame_index: int) -> None:
    """Draw the stop line, and each vehicle: neutral until it enters, then colored by the rule's verdict."""
    thickness = max(1, round(image.shape[0] / 540))
    cv2.polylines(image, [np.array(clip["stop_line"], np.int32)], False, (0, 255, 0), thickness, cv2.LINE_AA)
    for track_id, row in present:
        record = records_by_track.get(track_id)
        entered = record is not None and frame_index >= record["frame"]
        color = verdict_color(record["verdict"]) if entered else (235, 235, 235)
        flagged = entered and record["verdict"] == CANDIDATE
        x1, y1, x2, y2 = (int(round(v)) for v in row[1:5])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness * 2 if flagged else thickness)
        label = f"{track_id} RED LIGHT?" if flagged else track_id
        cv2.putText(image, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, max(1, thickness), cv2.LINE_AA)


def draw_banner(image: np.ndarray, text: str) -> None:
    """A line of text in a dark band at the top of the frame."""
    scale = max(0.5, 0.8 * image.shape[0] / 1080)
    font_thickness = max(1, round(scale * 2.5))
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, font_thickness)
    cv2.rectangle(image, (0, 0), (width + 16, height + 16), (0, 0, 0), -1)
    cv2.putText(image, text, (8, height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


def write_result_video(video_path: Path, output_path: Path, fps: float, clips: dict, tracks: dict, states: dict, records: dict) -> None:
    """Write the frames of each clip with the stop line, the signal state, each vehicle's verdict, and the candidates so far."""
    flagged = 0
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for name, clip in clips.items():
            by_track = {r["track_id"]: r for r in records[name]}
            present: dict[int, list] = {}
            for track_id, rows in tracks[name].items():
                for row in rows:
                    present.setdefault(int(row[0]), []).append((track_id, row))
            entries = sorted(r["frame"] for r in records[name] if r["verdict"] == CANDIDATE)
            for record in iter_frames(video_path, clip["start"], clip["end"]):
                image, k = record["image"], record["frame_index"]
                draw_clip_frame(image, clip, present.get(k, []), by_track, k)
                so_far = flagged + sum(entry <= k for entry in entries)
                draw_banner(image, f"through: {states[name]['through'].get(k, '?')}  left: {states[name]['left'].get(k, '?')}  candidates: {so_far} (pending review)")
                write_frame(image)
            flagged += len(entries)


def detect_red_light_video(video_path: str | Path, output_folder_path: str | Path, scene_path: str | Path, model=None) -> dict:
    """Find red-light candidates in a video, from start to finish.

    ``scene_path`` is the scene configuration of this camera: its stop line and signal positions
    cannot be read from the video. Only the frame ranges of its clips are analysed and saved. In
    ``output_folder_path`` this saves:

    - ``<video name>_output.mp4``: those frames, showing the stop line, the signal state and each verdict;
    - ``<video name>_violation_clips/``: one clip per candidate with original pixels, and ``manifest.json``;
    - ``<video name>_events.json``: the candidates, all ``pending``, and every crossing record.

    Returns these paths with the events and records. Earlier files of these names are replaced.
    """
    video_path, folder = Path(video_path), Path(output_folder_path)
    meta = read_video_metadata(video_path)
    fps, width, height = meta["fps"], meta["width"], meta["height"]
    scene = load_red_light_scene(scene_path, width, height)
    detector = model if model is not None else load_detector()
    clips = scene["clips"]
    video_id = video_path.stem

    tracks, states, records, events = {}, {}, {}, []
    for name, clip in clips.items():
        tracks[name], states[name] = track_and_read_clip(video_path, clip, detector)
        records[name] = evaluate_clip(tracks[name], clip, states[name], scene["lane_direction_deg"], width, height, fps)
        events += [build_event(r, name, video_id, fps, clip["start"]) for r in records[name] if r["verdict"] == CANDIDATE]

    output_video = output_video_path(video_path, folder)
    write_result_video(video_path, output_video, fps, clips, tracks, states, records)
    manifest = write_event_clips(
        video_path, events,
        lambda e: tracks[e["clip"]][e["track_id"]],
        lambda e: (clips[e["clip"]]["start"], clips[e["clip"]]["end"]),
        folder / f"{video_id}_violation_clips",
    )
    events_path = folder / f"{video_id}_events.json"
    events_path.write_text(json.dumps({
        "video_id": video_id,
        "scene_id": scene["scene_id"],
        "rule": RULE,
        "tracking_session": f"{video_id}/{TRACKING_SESSION}",
        "signal_source": "camera",
        "events": events,
        "crossings": records,
    }, indent=1))
    return {"output_video": output_video, "clips_manifest": manifest, "events_file": events_path, "events": events, "crossings": records}
