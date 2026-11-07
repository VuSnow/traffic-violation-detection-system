"""Command line.

- ``python -m tvd.cli track --video <file> --config <yaml> --output <json>``: detect and track.
- ``python -m tvd.cli check --video <file> --camera <yaml> --output <json>``: apply the camera's rules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tvd.camera import load_camera, load_rule_defaults
from tvd.check import check_frame_size, check_video
from tvd.config import ConfigError, PerceptionConfig, load_config
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
from tvd.perception import track_video
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader

PERCEPTION = Path("configs/tvd/perception.yaml")
RULE_DEFAULTS = Path("configs/tvd/rules.yaml")


def tracks_as_json(history: TrackHistory) -> list[dict]:
    """Each track with its class and its points as ``[frame, x1, y1, x2, y2, confidence]``."""
    return [
        {
            "track_id": track.track_id,
            "class_name": track.class_name,
            "points": [[p.frame_index, *p.box.as_list(), p.confidence] for p in track.points],
        }
        for track in history.tracks.values()
    ]


def run_track(video: Path, config_path: Path, output: Path) -> dict:
    config = load_config(config_path, PerceptionConfig)
    with VideoReader(video) as reader:  # A missing or unreadable video fails here, before any model is loaded.
        detector = build_detector(config.detector)
        tracker = build_tracker(config.tracker)
        start = time.perf_counter()
        history = track_video(reader, detector, tracker, set(config.track_classes))
    summary = {
        "video": video.name,
        "detector": config.detector.type,
        "tracker": config.tracker.type,
        "frames": history.frames_seen,
        "tracks": len(history.tracks),
        "seconds": round(time.perf_counter() - start, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({**summary, "track_list": tracks_as_json(history)}))
    return summary


def run_check(video: Path, camera_path: Path, output: Path, perception_path: Path = PERCEPTION, defaults_path: Path = RULE_DEFAULTS) -> dict:
    """Every settings file is checked, and the video opened, before any model is loaded."""
    perception = load_config(perception_path, PerceptionConfig)
    camera = load_camera(camera_path)
    rules = build_rules(camera, load_rule_defaults(defaults_path))
    motion_estimator = build_motion_estimator(camera.motion)
    with VideoReader(video) as reader:
        check_frame_size(reader, camera)
        detector = build_detector(perception.detector)
        tracker = build_tracker(perception.tracker)
        result = check_video(reader, camera, detector, tracker, set(perception.track_classes), motion_estimator, rules)
    data = result.as_json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=1))
    return data


def check_summary(data: dict) -> str:
    lines = [f"{data['video']}: {data['frames']} frames, {data['tracks']} tracks, camera {data['camera_id']}, {data['seconds']} s"]
    for name, rule in data["rules"].items():
        verdicts = ", ".join(f"{verdict} {count}" for verdict, count in sorted(rule["verdicts"].items()))
        lines.append(f"  {name}: {len(rule['findings'])} candidates for review; {verdicts or 'no tracks judged'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tvd")
    commands = parser.add_subparsers(dest="command", required=True)
    track = commands.add_parser("track", help="Detect and track the vehicles of a video")
    track.add_argument("--video", type=Path, required=True)
    track.add_argument("--config", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="Apply a camera's rules to a video of that camera")
    check.add_argument("--video", type=Path, required=True)
    check.add_argument("--camera", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--perception", type=Path, default=PERCEPTION)
    check.add_argument("--rules", type=Path, default=RULE_DEFAULTS)
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            summary = run_track(args.video, args.config, args.output)
            print(f"{summary['video']}: {summary['frames']} frames, {summary['tracks']} tracks, detector {summary['detector']}, tracker {summary['tracker']}, {summary['seconds']} s")
        else:
            print(check_summary(run_check(args.video, args.camera, args.output, args.perception, args.rules)))
    except (ConfigError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
