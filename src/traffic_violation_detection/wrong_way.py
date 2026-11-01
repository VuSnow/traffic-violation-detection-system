"""Wrong-way rule, developed in Notebook 05. Its output is a candidate for human review, never a verdict.

A track produces a candidate if, in some 2-second window, all of these hold:

- C1: the window's midpoint lies in a region with an allowed direction;
- C2: its net displacement points against that direction, within the region's tolerance;
- C3: the opposing distance is at least one box height;
- C4: the top edge of the box also moves against the direction, so a box that only grows does not count;
- C5: enough of the window is observed, and no box touches the image border.

Small functions come first. ``detect_wrong_way_video`` at the end joins them into one call that goes
from a video to a copy showing the result, one clip per candidate, and a file of events.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from traffic_violation_detection.camera import estimate_camera_motion
from traffic_violation_detection.detection import OUTPUT_CRF, load_detector
from traffic_violation_detection.evidence import write_event_clips
from traffic_violation_detection.scene import allowed_direction_unit, frame_to_reference, load_scene_config, point_in_polygon, reference_to_frame
from traffic_violation_detection.tracking import TRACKING_SESSION, collect_tracks, majority_class
from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer

RULE = {
    "name": "wrong_way",
    "version": 1,
    "window_seconds": 2.0,          # chosen by the sweep in Notebook 05
    "smoothing_seconds": 0.15,
    "merge_gap_seconds": 0.25,
    "min_opposing_heights": 1.0,    # C3
    "min_top_edge_share": 0.25,     # C4
    "min_observed_fraction": 0.8,   # C5
}
BORDER_MARGIN = 3  # px

CANDIDATE = "candidate"
NO_VIOLATION = "no_violation"
NOT_EVALUABLE = "not_evaluable"
NOT_IN_REGION = "not_in_directional_region"

STATUS_COLORS_BGR = {
    CANDIDATE: (40, 40, 230),
    NO_VIOLATION: (80, 200, 80),
    NOT_EVALUABLE: (170, 170, 170),
    NOT_IN_REGION: (235, 235, 235),
}


def rule_for_fps(fps: float, rule: dict = RULE) -> dict:
    """The rule with its durations in frames for a video's frame rate. The median filter needs an odd size."""
    return {
        **rule,
        "window_frames": round(rule["window_seconds"] * fps),
        "smoothing_frames": round(rule["smoothing_seconds"] * fps) | 1,
        "merge_gap_frames": round(rule["merge_gap_seconds"] * fps),
    }


def build_track(track_id: str, rows: list, params: np.ndarray, width: int, height: int) -> dict:
    """One vehicle's measurements: its bottom and top centers in reference coordinates, its size, and border contact."""
    data = np.array(rows, dtype=np.float64)
    frames = data[:, 0].astype(int)
    boxes = data[:, 1:5]
    centers = (boxes[:, 0] + boxes[:, 2]) / 2
    return {
        "track_id": track_id,
        "class_name": majority_class(rows),
        "frames": frames,
        "bottom_ref": np.vstack([frame_to_reference(params[f], [[x, y]]) for f, x, y in zip(frames, centers, boxes[:, 3])]),
        "top_ref": np.vstack([frame_to_reference(params[f], [[x, y]]) for f, x, y in zip(frames, centers, boxes[:, 1])]),
        "heights": boxes[:, 3] - boxes[:, 1],
        "clipped": (boxes[:, 0] <= BORDER_MARGIN) | (boxes[:, 1] <= BORDER_MARGIN) | (boxes[:, 2] >= width - BORDER_MARGIN) | (boxes[:, 3] >= height - BORDER_MARGIN),
    }


def track_windows(track: dict, scene: dict, rule: dict) -> tuple[list[dict], bool]:
    """All windows whose midpoint is in a directional region, with their measurements. No thresholds apply here."""
    frames = track["frames"]
    kernel = (rule["smoothing_frames"], 1)
    bottom = median_filter(track["bottom_ref"], size=kernel, mode="nearest")
    top = median_filter(track["top_ref"], size=kernel, mode="nearest")
    regions = [r for r in scene["regions"] if r["allowed_direction"]]
    seen_in_region = any(point_in_polygon(r["polygon"], p) for r in regions for p in bottom[::5])
    windows = []
    for i, start_frame in enumerate(frames):
        j = int(np.searchsorted(frames, start_frame + rule["window_frames"]))
        if j >= len(frames) or frames[j] != start_frame + rule["window_frames"]:
            continue
        midpoint = (bottom[i] + bottom[j]) / 2
        region = next((r for r in regions if point_in_polygon(r["polygon"], midpoint)), None)
        if region is None:
            continue
        allowed = allowed_direction_unit(region, midpoint, scene["vanishing_point"])
        displacement = bottom[j] - bottom[i]
        distance = float(np.linalg.norm(displacement))
        along = float(-(displacement @ allowed))
        box_height = float(np.median(track["heights"][i:j + 1]))
        windows.append({
            "start_frame": int(frames[i]),
            "end_frame": int(frames[j]),
            "region_id": region["id"],
            "opposing_px": along,
            "opposing_heights": along / box_height,
            "top_opposing_px": float(-((top[j] - top[i]) @ allowed)),
            "angle_to_opposite_deg": float(np.degrees(np.arccos(np.clip(along / distance, -1, 1)))) if distance > 0 else 180.0,
            "tolerance_deg": region["direction_tolerance_deg"],
            "observed_fraction": (j - i + 1) / (rule["window_frames"] + 1),
            "clipped": bool(track["clipped"][i:j + 1].any()),
        })
    return windows, seen_in_region


def usable(window: dict, rule: dict) -> bool:
    """C5: the window can be trusted."""
    return window["observed_fraction"] >= rule["min_observed_fraction"] and not window["clipped"]


def failed_conditions(window: dict, rule: dict) -> list[str]:
    """Which conditions a window fails; an empty list means it passes."""
    failed = []
    if window["angle_to_opposite_deg"] > window["tolerance_deg"]:
        failed.append("C2 direction")
    if window["opposing_heights"] < rule["min_opposing_heights"]:
        failed.append("C3 distance")
    if window["opposing_px"] <= 0 or window["top_opposing_px"] < rule["min_top_edge_share"] * window["opposing_px"]:
        failed.append("C4 top edge")
    if not usable(window, rule):
        failed.append("C5 quality")
    return failed


def merge_windows(passing: list[dict], rule: dict) -> list[list[dict]]:
    """Group overlapping passing windows of one region: each group is one event."""
    groups: list[list[dict]] = []
    for window in sorted(passing, key=lambda w: w["start_frame"]):
        if groups and window["region_id"] == groups[-1][-1]["region_id"] and window["start_frame"] <= groups[-1][-1]["end_frame"] + rule["merge_gap_frames"]:
            groups[-1].append(window)
        else:
            groups.append([window])
    return groups


def build_event(track: dict, group: list[dict], rule: dict, video_id: str, fps: float) -> dict:
    """One candidate event. Its status is always pending, and its id never reuses the track id alone."""
    first, last = group[0], group[-1]
    best = max(group, key=lambda w: w["opposing_heights"])
    start, trigger, end = first["start_frame"], first["end_frame"], last["end_frame"]
    return {
        "event_id": f"evt-{video_id}-{rule['name']}-f{start:05d}-t{track['track_id']}",
        "video_id": video_id,
        "violation_type": rule["name"],
        "tracking_session": f"{video_id}/{TRACKING_SESSION}",
        "track_id": track["track_id"],
        "class_name": track["class_name"],
        "region_id": first["region_id"],
        "start_frame": start,
        "trigger_frame": trigger,
        "end_frame": end,
        "start_time_s": round(start / fps, 3),
        "trigger_time_s": round(trigger / fps, 3),
        "end_time_s": round(end / fps, 3),
        "rule": {k: rule[k] for k in ("name", "version", "window_seconds", "min_opposing_heights", "min_top_edge_share", "min_observed_fraction")},
        "why": (f"Moved against the allowed direction of {first['region_id']} by {best['opposing_heights']:.2f} box heights "
                f"in about {rule['window_seconds']:.1f} s, {best['angle_to_opposite_deg']:.0f} degrees from the opposite direction."),
        "evidence": {
            "opposing_heights": round(best["opposing_heights"], 3),
            "opposing_px": round(best["opposing_px"], 1),
            "top_opposing_px": round(best["top_opposing_px"], 1),
            "angle_to_opposite_deg": round(best["angle_to_opposite_deg"], 1),
            "evidence_frames": [start, trigger, end],
        },
        "license_plate": None,
        "review_status": "pending",
    }


def evaluate_track(track: dict, scene: dict, rule: dict, video_id: str = "video", fps: float = 30.0) -> dict:
    """Judge one track. ``not_evaluable`` is kept apart: "we could not check" differs from "we found nothing"."""
    windows, seen_in_region = track_windows(track, scene, rule)
    evaluable = [w for w in windows if usable(w, rule)]
    passing = [w for w in evaluable if not failed_conditions(w, rule)]
    if passing:
        status = CANDIDATE
    elif evaluable:
        status = NO_VIOLATION
    elif seen_in_region or windows:
        status = NOT_EVALUABLE
    else:
        status = NOT_IN_REGION
    return {
        "track_id": track["track_id"],
        "status": status,
        "evaluable_windows": len(evaluable),
        "events": [build_event(track, group, rule, video_id, fps) for group in merge_windows(passing, rule)],
    }


def tracks_by_frame(tracks: dict[str, list]) -> dict[int, list[tuple[str, list]]]:
    """For each frame, the tracks present in it and their rows."""
    frames: dict[int, list] = {}
    for track_id, rows in tracks.items():
        for row in rows:
            frames.setdefault(int(row[0]), []).append((track_id, row))
    return frames


def draw_regions(image: np.ndarray, scene: dict, params) -> None:
    """Draw each region with an allowed direction, and an arrow for that direction, on a BGR frame, in place."""
    thickness = max(1, round(image.shape[0] / 540))
    for region in scene["regions"]:
        if not region["allowed_direction"]:
            continue
        polygon = np.array(region["polygon"], dtype=np.float64)
        cv2.polylines(image, [np.round(reference_to_frame(params, polygon)).astype(np.int32)], True, (255, 200, 0), thickness, cv2.LINE_AA)
        center = polygon.mean(axis=0)
        unit = allowed_direction_unit(region, center, scene["vanishing_point"])
        arrow = np.round(reference_to_frame(params, [center, center + unit * 60])).astype(int)
        cv2.arrowedLine(image, tuple(int(v) for v in arrow[0]), tuple(int(v) for v in arrow[1]), (255, 200, 0), thickness + 1, cv2.LINE_AA, tipLength=0.3)


def draw_track_statuses(image: np.ndarray, present: list[tuple[str, list]], statuses: dict[str, str], events_by_track: dict[str, list], frame_index: int) -> None:
    """Draw each vehicle colored by its result. A candidate is red only during the frames of its event."""
    thickness = max(1, round(image.shape[0] / 540))
    for track_id, row in present:
        status = statuses[track_id]
        in_event = any(e["start_frame"] <= frame_index <= e["end_frame"] for e in events_by_track.get(track_id, []))
        shown = CANDIDATE if in_event else (NOT_EVALUABLE if status == CANDIDATE else status)
        color = STATUS_COLORS_BGR[shown]
        x1, y1, x2, y2 = (int(round(v)) for v in row[1:5])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness * 2 if in_event else thickness)
        label = f"{track_id} WRONG WAY?" if in_event else track_id
        cv2.putText(image, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, max(1, thickness), cv2.LINE_AA)


def draw_banner(image: np.ndarray, text: str) -> None:
    """A line of text in a dark band at the top of the frame."""
    scale = max(0.5, 0.8 * image.shape[0] / 1080)
    thickness = max(1, round(scale * 2.5))
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (0, 0), (width + 16, height + 16), (0, 0, 0), -1)
    cv2.putText(image, text, (8, height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def write_result_video(video_path: Path, output_path: Path, fps: float, scene: dict, params: np.ndarray, tracks: dict, statuses: dict, events: list[dict]) -> None:
    """Write the copy of the video showing the scene, each vehicle's result, and the candidates so far."""
    present = tracks_by_frame(tracks)
    events_by_track: dict[str, list] = {}
    for event in events:
        events_by_track.setdefault(event["track_id"], []).append(event)
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for record in iter_frames(video_path):
            image, k = record["image"], record["frame_index"]
            draw_regions(image, scene, params[k])
            draw_track_statuses(image, present.get(k, []), statuses, events_by_track, k)
            draw_banner(image, f"Wrong-way candidates so far: {sum(e['trigger_frame'] <= k for e in events)} (pending review)")
            write_frame(image)


def detect_wrong_way_video(video_path: str | Path, output_folder_path: str | Path, scene_path: str | Path, model=None) -> dict:
    """Find wrong-way candidates in a video, from start to finish.

    ``scene_path`` is the scene configuration of this camera: its regions and allowed directions
    cannot be read from the video. In ``output_folder_path`` this saves:

    - ``<video name>_output.mp4``: a copy showing each vehicle's result;
    - ``<video name>_violation_clips/``: one clip per candidate with original pixels, and ``manifest.json``;
    - ``<video name>_events.json``: the candidates, all ``pending``, and the status of every track.

    Returns these paths with the events and statuses. Earlier files of these names are replaced.
    """
    video_path, folder = Path(video_path), Path(output_folder_path)
    meta = read_video_metadata(video_path)
    fps, width, height = meta["fps"], meta["width"], meta["height"]
    scene = load_scene_config(scene_path, width, height)
    detector = model if model is not None else load_detector()
    rule = rule_for_fps(fps)
    video_id = video_path.stem

    tracks, detections_by_frame = collect_tracks(video_path, detector)
    params = estimate_camera_motion(video_path, detections_by_frame, scene["coordinates"]["reference_frame_index"])
    results = {t: evaluate_track(build_track(t, rows, params, width, height), scene, rule, video_id, fps) for t, rows in tracks.items()}
    statuses = {t: r["status"] for t, r in results.items()}
    events = sorted((e for r in results.values() for e in r["events"]), key=lambda e: e["trigger_frame"])

    output_video = output_video_path(video_path, folder)
    write_result_video(video_path, output_video, fps, scene, params, tracks, statuses, events)
    frame_count = len(detections_by_frame)
    manifest = write_event_clips(video_path, events, lambda e: tracks[e["track_id"]], lambda e: (0, frame_count), folder / f"{video_id}_violation_clips")
    events_path = folder / f"{video_id}_events.json"
    events_path.write_text(json.dumps({
        "video_id": video_id,
        "scene_id": scene["scene_id"],
        "rule": rule,
        "tracking_session": f"{video_id}/{TRACKING_SESSION}",
        "status_counts": dict(Counter(statuses.values())),
        "candidates": events,
        "track_statuses": statuses,
    }, indent=1))
    return {"output_video": output_video, "clips_manifest": manifest, "events_file": events_path, "events": events, "track_statuses": statuses}
