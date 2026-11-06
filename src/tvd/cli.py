"""Command line: ``python -m tvd.cli track --video <file> --config <yaml> --output <json>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tvd.config import ConfigError, PerceptionConfig, load_config
from tvd.factory import build_detector, build_tracker
from tvd.perception import track_video
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tvd")
    commands = parser.add_subparsers(dest="command", required=True)
    track = commands.add_parser("track", help="Detect and track the vehicles of a video")
    track.add_argument("--video", type=Path, required=True)
    track.add_argument("--config", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_track(args.video, args.config, args.output)
    except (ConfigError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{summary['video']}: {summary['frames']} frames, {summary['tracks']} tracks, detector {summary['detector']}, tracker {summary['tracker']}, {summary['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
