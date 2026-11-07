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
