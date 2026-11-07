"""The notebooks' scene files and saved results, turned into what the tvd package takes, for tests that compare the two.

The saved results are in outputs/, which only the GPU server has: tests that need them skip elsewhere.
"""

import json
from pathlib import Path

from tvd.domain import Box, TrackedObject
from tvd.tracking.history import TrackHistory

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
RED_LIGHT_SCENE = ROOT / "configs" / "scenes" / "rlr_occasions.json"
RLR_VIDEO = ROOT / "data" / "raw" / "rlr_occasions.mp4"
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def wrong_way_scene(scene: dict) -> dict:
    """A notebook wrong-way scene as the ``scene`` of a camera file's ``wrong_way`` rule: only regions with a direction matter."""
    return {
        "vanishing_point": scene["vanishing_point"],
        "regions": [
            {"id": r["id"], "polygon": r["polygon"], "allowed_direction": r["allowed_direction"], "tolerance_deg": r["direction_tolerance_deg"]}
            for r in scene["regions"] if r["allowed_direction"]
        ],
    }


def red_light_section(scene: dict, clip_name: str) -> dict:
    """One clip of a notebook red-light scene as the ``red_light`` section of a camera file.

    The notebooks call the night reader "head"; here it is "glow", and each head is ``{"roi": [x1, y1, x2, y2]}``.
    """
    clip = scene["clips"][clip_name]
    signals = clip["signals"]
    lamps = signals["method"] == "lamps"
    options = {"half": signals["lamp_half"]} if lamps and "lamp_half" in signals else {}
    head = (lambda h: dict(h)) if lamps else (lambda h: {"roi": list(h)})
    return {
        "scene": {
            "stop_line": clip["stop_line"],
            "lane_dividers": clip["lane_dividers"],
            "lane_direction_deg": scene["lane_direction_deg"],
            "traffic_direction": clip.get("traffic_direction", "away_from_camera"),
            "evaluable_x_max": clip.get("evaluable_x_max"),
            "signal_source": scene.get("signal_source", "camera"),
            "signals": {
                "reader": "lamps" if lamps else "glow",
                "options": options,
                "through": [head(h) for h in signals["through"]],
                "left": [head(h) for h in signals["left"]],
            },
        },
        "settings": {k: clip[k] for k in ("entry_offset", "pass_margin") if k in clip},
    }


def history_from_rows(tracks: dict[str, list]) -> TrackHistory:
    """A ``TrackHistory`` from the notebooks' rows of ``frame, x1, y1, x2, y2, class_id, confidence``, in the same order of tracks."""
    by_frame: dict[int, list[TrackedObject]] = {}
    for track_id, rows in tracks.items():
        for frame, x1, y1, x2, y2, class_id, confidence in rows:
            by_frame.setdefault(int(frame), []).append(TrackedObject(int(track_id), Box(x1, y1, x2, y2), int(class_id), VEHICLE_CLASSES[int(class_id)], confidence))
    history = TrackHistory()
    for frame in sorted(by_frame):
        history.add(frame, by_frame[frame])
    return history


def load_json(path: Path):
    return json.loads(path.read_text())


MODIFIED_VIDEO = ROOT / "data" / "traffic_video_modified.mp4"
NOTEBOOK_08 = OUTPUTS / "08_end_to_end_baseline" / "traffic_video_modified"
NOTEBOOK_08_CLIPS = NOTEBOOK_08 / "violations" / "traffic_video_modified_violation_clips"


def notebook_08_clips() -> list[dict]:
    """The manifest entries of the two red-light clips that Notebook 08 cut from traffic_video_modified.mp4."""
    return load_json(NOTEBOOK_08_CLIPS / "manifest.json")["clips"]


def finding_from_clip(entry: dict):
    """A red-light finding with the vehicle boxes and trigger of a Notebook 08 clip: enough to cut the same clip again."""
    from tvd.rules import Finding
    from tvd.tracking.history import TrackPoint

    start = entry["source_start_frame"]
    points = tuple(TrackPoint(start + b["frame_in_clip"], Box(*b["box_xyxy"]), 0.9, entry["vehicle_class"]) for b in entry["vehicle_boxes"])
    trigger = start + entry["trigger_frame_in_clip"]
    return Finding("red_light", entry["event_id"].rsplit("-t", 1)[1], entry["vehicle_class"], points[0].frame_index, trigger, points[-1].frame_index, "", {}, points)


NOTEBOOK_08_PLATES = NOTEBOOK_08 / "plates"


def notebook_08_plates() -> dict[str, dict]:
    """Notebook 08's plate result for each clip, by event id. They hold real plate numbers: never print them."""
    return {e["event_id"]: e for e in load_json(NOTEBOOK_08_PLATES / "plates.json")["events"]}


def plate_read_from_json(read: dict):
    """A read that the notebooks saved. Its image was not saved: it is replaced by one black pixel."""
    import numpy as np

    from tvd.plates import PlateRead

    keys = ("frame", "text", "mean_prob", "min_prob", "has_gap", "detection_confidence", "plate_width", "plate_height")
    return PlateRead(*(read[k] for k in keys), Box(*read["plate_box_xyxy"]), read["rejected"], np.zeros((1, 1, 3), np.uint8))


NOTEBOOK_08_EVIDENCE = NOTEBOOK_08 / "events" / "evidence"


def clip_from_entry(entry: dict):
    """A Notebook 08 clip, as tvd describes a clip."""
    from tvd.evidence import Clip

    start = entry["source_start_frame"]
    boxes = {b["frame_in_clip"]: Box(*b["box_xyxy"]) for b in entry["vehicle_boxes"]}
    return Clip(NOTEBOOK_08_CLIPS / entry["clip_file"], start, entry["source_end_frame"], entry["fps"], entry["width"], entry["height"], start + entry["trigger_frame_in_clip"], boxes)


def notebook_08_records() -> dict[str, dict]:
    """Notebook 08's records, by event id. They hold real plate numbers: never print them."""
    return {r["event_id"]: r for r in load_json(NOTEBOOK_08 / "events" / "traffic_video_modified_review_events.json")["events"]}


NOTEBOOK_07_REVERSED = OUTPUTS / "07_violation_event_generation" / "wrong_way_control" / "video-test-reversed_review_events.json"
REVERSED_CANDIDATES = OUTPUTS / "05_violation_detection" / "time_reversed_control" / "video-test-reversed_events.json"


def finding_from_event(event: dict, points: tuple = ()):
    """A finding saying what a notebook's candidate event says. A red-light event has no start or end: its points give them."""
    from tvd.rules import Finding

    evidence = {**({"region_id": event["region_id"]} if "region_id" in event else {}), **event["evidence"]}
    if event.get("signal_source"):
        evidence["signal_source"] = event["signal_source"]
    start = event.get("start_frame", points[0].frame_index if points else event["trigger_frame"])
    end = event.get("end_frame", points[-1].frame_index if points else event["trigger_frame"])
    return Finding(event["violation_type"], event["track_id"], event["class_name"], start, event["trigger_frame"], end, event["why"], evidence, points)


NOTEBOOK_08_RLR = OUTPUTS / "08_end_to_end_baseline" / "rlr_occasions"


def cut_losslessly(source: Path, start: int, end: int, path: Path) -> Path:
    """Frames ``start`` to ``end`` (not included) of a video, in a file of their own that decodes to exactly the same pixels (FFV1)."""
    import subprocess

    from tvd.video import VideoReader, find_ffmpeg

    with VideoReader(source) as reader:
        size, fps = f"{reader.width}x{reader.height}", reader.fps
        command = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", size, "-r", str(fps), "-i", "-", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame in reader.frames(start, end):
            encoder.stdin.write(frame.image.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg could not write {path}")
    return path
