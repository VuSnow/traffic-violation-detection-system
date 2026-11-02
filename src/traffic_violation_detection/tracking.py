"""Multi-object tracking on a whole video, with the baseline chosen in Notebook 03: OC-SORT with low-score association.

Small functions come first, one for each step. ``track_video`` at the end joins them into one call
that goes from a video to a copy showing each vehicle's identity and recent path.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from traffic_violation_detection.detection import OUTPUT_CRF, detect, load_detector
from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer

VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRACKING_SESSION = "ocsort-byte"
TRAIL_FRAMES = 60  # How much of a vehicle's recent path is drawn.


def create_tracker():
    """A new tracker. Use one per continuous shot, because a vehicle cannot continue across a cut."""
    try:
        import boxmot
        import boxmot.utils  # noqa: F401  Imported so that the logger below exists.
    except ImportError as error:
        raise RuntimeError('Tracking needs BoxMOT: pip install -e ".[tracking]" && pip install --no-deps boxmot==26.0.0') from error
    logging.getLogger("boxmot").setLevel(logging.WARNING)
    return boxmot.OcSort(boxmot.OcSortConfig(use_byte=True))


def vehicle_detections(detections: list[dict]) -> np.ndarray:
    """The vehicles among a frame's detections, as tracker input: rows of ``x1, y1, x2, y2, confidence, class_id``."""
    rows = [[*d["bbox_xyxy"], d["confidence"], d["class_id"]] for d in detections if d["class_id"] in VEHICLE_CLASSES]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def update_tracker(tracker, detections: list[dict], image: np.ndarray) -> list[dict]:
    """Advance the tracker by one frame and return the vehicles it is tracking in that frame."""
    rows = vehicle_detections(detections)
    output = tracker.update(rows, image)
    tracked = []
    for row in output if len(output) else []:
        class_id = int(rows[int(row[7]), 5])  # The class of the detection matched in this frame.
        tracked.append({
            "track_id": int(row[4]),
            "class_id": class_id,
            "class_name": VEHICLE_CLASSES[class_id],
            "confidence": round(float(row[5]), 4),
            "bbox_xyxy": [round(float(v), 2) for v in row[:4]],
        })
    return tracked


def track_frames(video_path: str | Path, detector, start: int = 0, end: int | None = None) -> Iterator[tuple[dict, list[dict], list[dict]]]:
    """Yield ``(frame record, detections, tracked vehicles)`` for each frame, with a new tracker for this range."""
    tracker = create_tracker()
    for record in iter_frames(video_path, start, end):
        detections = detect(detector, record["image"])
        yield record, detections, update_tracker(tracker, detections, record["image"])


def add_to_tracks(tracks: dict[str, list], frame_index: int, tracked: list[dict]) -> None:
    """Append one frame to ``tracks``, which maps a track id to rows of ``frame, x1, y1, x2, y2, class_id, confidence``."""
    for t in tracked:
        tracks.setdefault(str(t["track_id"]), []).append([frame_index, *t["bbox_xyxy"], t["class_id"], t["confidence"]])


def collect_tracks(video_path: str | Path, detector, start: int = 0, end: int | None = None) -> tuple[dict[str, list], list[list[dict]]]:
    """Track a range of a video. Returns the rows of each track, and the detections of each frame in the range."""
    tracks: dict[str, list] = {}
    detections_by_frame = []
    for record, detections, tracked in track_frames(video_path, detector, start, end):
        add_to_tracks(tracks, record["frame_index"], tracked)
        detections_by_frame.append(detections)
    return tracks, detections_by_frame


def join_track_fragments(tracks: dict[str, list], max_gap_frames: int, max_jump_heights: float, max_size_ratio: float, box_of=None) -> dict[str, list]:
    """Join tracks that are plainly one vehicle under two ids: one ends where the other begins, at the same size.

    The limits are strict on purpose. Joining two vehicles could give one of them the other's plate,
    while leaving a vehicle in two parts only costs a missed case. ``box_of(row)`` gives a row's box,
    in reference coordinates if the camera moves. A joined track's id is ``"<first>+<second>"``.
    """
    box_of = box_of or (lambda row: row[1:5])

    def bottom_and_height(row: list) -> tuple[float, float, float]:
        x1, y1, x2, y2 = box_of(row)
        return (x1 + x2) / 2, y2, y2 - y1

    def continuation(first: list, second: list) -> float | None:
        # How far the second track starts from the end of the first, in box heights, or None if it does not continue it.
        if not (second[0][0] > first[0][0] and second[-1][0] > first[-1][0] and abs(second[0][0] - first[-1][0]) <= max_gap_frames):
            return None
        x1, y1, height1 = bottom_and_height(first[-1])
        x2, y2, height2 = bottom_and_height(second[0])
        jump = float(np.hypot(x2 - x1, y2 - y1)) / max(height1, 1)
        return jump if jump <= max_jump_heights and max(height1 / height2, height2 / height1) <= max_size_ratio else None

    joined = {track_id: list(rows) for track_id, rows in tracks.items()}
    while True:
        pairs = [(continuation(joined[a], joined[b]), a, b) for a in joined for b in joined if a != b]
        pairs = [p for p in pairs if p[0] is not None]
        if not pairs:
            return joined
        _, first, second = min(pairs)
        later = joined.pop(second)
        earlier = joined.pop(first)
        joined[f"{first}+{second}"] = [row for row in earlier if row[0] < later[0][0]] + later


def majority_class(rows: list) -> str:
    """The class most often given to a track, since a single frame's class can be wrong."""
    class_id = Counter(int(row[5]) for row in rows).most_common(1)[0][0]
    return VEHICLE_CLASSES.get(class_id, str(class_id))


def track_color(track_id: int) -> tuple[int, int, int]:
    """A color that stays the same for a track throughout the video."""
    return tuple(int(c) for c in np.random.default_rng(track_id).integers(60, 255, 3))


def update_trails(trails: dict[int, list], tracked: list[dict], length: int = TRAIL_FRAMES) -> None:
    """Add each vehicle's current position, the bottom center of its box, to its trail."""
    for t in tracked:
        x1, _, x2, y2 = t["bbox_xyxy"]
        trail = trails.setdefault(t["track_id"], [])
        trail.append(((x1 + x2) / 2, y2))
        del trail[:-length]


def draw_tracks(image: np.ndarray, tracked: list[dict], trails: dict[int, list] | None = None) -> None:
    """Draw each vehicle's box, id and trail on a BGR frame, in place. At 1080p this matches Notebook 03."""
    height = image.shape[0]
    thickness = max(1, round(height / 540))
    font_scale = max(0.5, 0.6 * height / 1080)
    font_thickness = max(1, round(font_scale * 3.4))
    for t in tracked:
        color = track_color(t["track_id"])
        x1, y1, x2, y2 = (int(round(v)) for v in t["bbox_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = str(t["track_id"])
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, (x1, max(0, y1 - text_height - 6)), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        if trails and t["track_id"] in trails:
            cv2.polylines(image, [np.array(trails[t["track_id"]], dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def track_video(video_path: str | Path, output_folder_path: str | Path, model=None) -> Path:
    """Track the vehicles of a video and save a copy showing each one's box, id and recent path.

    The output is ``<output_folder_path>/<video name>_output.mp4``, at the size and frame rate of the
    input. It replaces an earlier file of that name, so use one folder per step. Returns its path.
    """
    video_path = Path(video_path)
    fps = read_video_metadata(video_path)["fps"]
    detector = model if model is not None else load_detector()
    output_path = output_video_path(video_path, output_folder_path)
    trails: dict[int, list] = {}
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for record, _, tracked in track_frames(video_path, detector):
            update_trails(trails, tracked)
            draw_tracks(record["image"], tracked, trails)
            write_frame(record["image"])
    return output_path
