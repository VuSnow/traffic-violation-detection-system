"""Command line.

- ``python -m tvd.cli track --video <file> --config <yaml> --output <json>``: detect and track.
- ``python -m tvd.cli check --video <file> --camera <yaml> --output <json>``: apply the camera's rules.
- ``python -m tvd.cli review --video <file> --camera <yaml> --recorded-at <time> --output <folder>``: and write
  each candidate's evidence and record, for a person to review.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tvd.camera import load_camera, load_rule_defaults
from tvd.check import check_frame_size, check_video
from tvd.config import PerceptionConfig, file_version, load_config
from tvd.evidence import load_evidence_settings
from tvd.factory import build_detector, build_motion_estimator, build_plate_models, build_rules, build_tracker
from tvd.perception import track_video
from tvd.plates import load_plates_config
from tvd.records import RecordContext, parse_recorded_at
from tvd.review import ReviewResult, prepare_folder, review_check
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader

PERCEPTION = Path("configs/tvd/perception.yaml")
RULE_DEFAULTS = Path("configs/tvd/rules.yaml")
PLATES = Path("configs/tvd/plates.yaml")
EVIDENCE = Path("configs/tvd/evidence.yaml")


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


def run_review(
    video: Path, camera_path: Path, recorded_at: str, output: Path, perception_path: Path = PERCEPTION,
    defaults_path: Path = RULE_DEFAULTS, plates_path: Path = PLATES, evidence_path: Path = EVIDENCE,
) -> ReviewResult:
    """Every settings file, the time and the output folder are checked, and the video opened, before any model is loaded.

    Plate models that cannot start do not stop the review: its records say that plates were not read, and why.
    """
    moment = parse_recorded_at(recorded_at)
    perception = load_config(perception_path, PerceptionConfig)
    camera = load_camera(camera_path)
    rules = build_rules(camera, load_rule_defaults(defaults_path))
    motion_estimator = build_motion_estimator(camera.motion)
    plates = load_plates_config(plates_path)
    evidence = load_evidence_settings(evidence_path)
    files = {"perception": perception_path, "camera": camera_path, "rules": defaults_path, "plates": plates_path, "evidence": evidence_path}
    config_files = {name: file_version(path) for name, path in files.items()}
    with VideoReader(video) as reader:
        check_frame_size(reader, camera)
        prepare_folder(output)
        try:
            plate_models, unavailable = build_plate_models(plates.models), ""
        except RuntimeError as error:
            plate_models, unavailable = None, f"the plate models did not start: {error}"
        try:
            detector, tracker = build_detector(perception.detector), build_tracker(perception.tracker)
            check = check_video(reader, camera, detector, tracker, set(perception.track_classes), motion_estimator, rules)
            settings = {
                "perception": perception.model_dump(), "rules": {name: r.settings for name, r in check.results.items()},
                "plates": plates.model_dump(), "evidence": evidence.model_dump(),
            }
            context = RecordContext(
                video.stem, reader.fps, reader.width, reader.height, camera.camera_id, moment, f"{video.stem}/{perception.tracker.type}", settings, config_files,
            )
            return review_check(reader, check, context, plate_models, plates.reading, evidence, output, unavailable or "no plate models")
        finally:
            if plate_models is not None:
                plate_models.close()


def review_summary(result: ReviewResult) -> str:
    """Counts only: plate numbers are never printed."""
    plates = ", ".join(f"{status} {count}" for status, count in sorted(result.summary()["plates"].items())) or "none"
    lines = [
        check_summary(result.check.as_json()),
        f"  {len(result.records)} records for review in {result.folder} (duplicates dropped: {len(result.duplicates)}); plates: {plates}",
    ]
    return "\n".join(lines + [f"  failure: {failure}" for failure in result.failures])


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
    review = commands.add_parser("review", help="Apply a camera's rules to a video, and write each candidate's evidence and record")
    review.add_argument("--video", type=Path, required=True)
    review.add_argument("--camera", type=Path, required=True)
    review.add_argument("--recorded-at", required=True, help="The time of the first frame, with its time zone: 2026-11-07T08:00:00+07:00")
    review.add_argument("--output", type=Path, required=True, help="A new or empty folder")
    review.add_argument("--perception", type=Path, default=PERCEPTION)
    review.add_argument("--rules", type=Path, default=RULE_DEFAULTS)
    review.add_argument("--plates", type=Path, default=PLATES)
    review.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            summary = run_track(args.video, args.config, args.output)
            print(f"{summary['video']}: {summary['frames']} frames, {summary['tracks']} tracks, detector {summary['detector']}, tracker {summary['tracker']}, {summary['seconds']} s")
        elif args.command == "check":
            print(check_summary(run_check(args.video, args.camera, args.output, args.perception, args.rules)))
        else:
            result = run_review(args.video, args.camera, args.recorded_at, args.output, args.perception, args.rules, args.plates, args.evidence)
            print(review_summary(result))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:  # ValueError covers ConfigError.
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
