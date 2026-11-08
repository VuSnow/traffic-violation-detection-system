"""Records of candidates, schema version 2 (``docs/event_schema.md``): one per candidate, for a person to review.

A record joins what the pass found (the finding), its evidence (the clip and its images), and its plate.
It is always ``pending``: nothing in the service confirms a violation. Reviewing is the backend's job.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tvd.evidence import Clip, EvidenceImage
from tvd.plates import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateResult
from tvd.rules import Finding

SCHEMA_VERSION = 2
PENDING = "pending"
NOT_ATTEMPTED = "not_attempted"
PLATE_STATUSES = (READ, UNCERTAIN, UNREADABLE, NO_PLATE, NOT_ATTEMPTED)
PLATE_ATTENTION = {UNCERTAIN: "plate_uncertain", UNREADABLE: "plate_unreadable", NO_PLATE: "plate_not_found", NOT_ATTEMPTED: "plate_not_attempted"}
REQUIRED_FIELDS = {
    "schema_version": int, "event_id": str, "camera_id": str, "video": dict, "violation_type": str, "track": dict, "time": dict, "rule": dict,
    "why": str, "signal_source": (str, type(None)), "measurements": dict, "evidence": dict, "license_plate": dict, "needs_attention": list,
    "settings": dict, "config_files": dict, "models": dict, "review": dict,
}


def parse_recorded_at(text: str) -> datetime:
    """The time of a video's first frame, in ISO 8601 with its time zone, such as ``2026-11-07T08:00:00+07:00``."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"recorded_at is not an ISO 8601 time: {text!r}") from error
    if moment.tzinfo is None:
        raise ValueError(f"recorded_at needs a time zone, such as 2026-11-07T08:00:00+07:00: {text!r}")
    return moment


def time_at(recorded_at: datetime, seconds: float) -> str:
    """The absolute time of a moment of the video, rounded to the millisecond as ``*_time_s`` are, in the time zone that was sent."""
    return (recorded_at + timedelta(milliseconds=round(seconds * 1000))).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RecordContext:
    """What every record of one video shares.

    ``settings`` are the effective settings, as ``{"perception": ..., "rules": {name: ...}, "plates": ..., "evidence": ...}``,
    ``config_files`` the version of each settings file that gave them, and ``models`` the models and software (``models_used``).
    """

    video_id: str
    fps: float
    width: int
    height: int
    camera_id: str
    recorded_at: datetime
    tracking_session: str
    settings: dict[str, Any]
    config_files: dict[str, str]
    models: dict[str, Any]


def event_id(video_id: str, finding: Finding) -> str:
    """Unique within a video: one rule fires once per track and frame. It never equals a track id."""
    return f"evt-{video_id}-{finding.rule}-f{finding.trigger_frame:05d}-t{finding.track_id}"


def plate_summary(result: PlateResult | None) -> dict[str, Any]:
    """What a record keeps of a plate result. Without one, plate reading was not attempted."""
    if result is None:
        return {"status": NOT_ATTEMPTED, "text": None, "votes": 0, "usable_reads": 0, "agreement": 0.0, "mean_prob": 0.0}
    return {"status": result.status, "text": result.text, "votes": result.votes, "usable_reads": result.usable_reads, "agreement": result.agreement, "mean_prob": result.mean_prob}


def attention_flags(finding: Finding, plate: dict[str, Any]) -> list[str]:
    """What a reviewer should look at closely, because the record alone cannot settle it."""
    flags = []
    if finding.evidence.get("signal_source") == "simulated_overlay":
        flags.append("simulated_signal")
    if finding.evidence.get("entry_estimated"):
        flags.append("entry_estimated")
    elif finding.evidence.get("entry_seen") is False:
        flags.append("entry_not_seen")
    if plate["status"] in PLATE_ATTENTION:
        flags.append(PLATE_ATTENTION[plate["status"]])
    return flags


def relative_to(path: Path, folder: Path) -> str:
    return Path(os.path.relpath(path, folder)).as_posix()


def build_record(
    context: RecordContext, finding: Finding, clip: Clip, images: list[EvidenceImage],
    plate: PlateResult | None, plate_image: Path | None, folder: Path,
) -> dict[str, Any]:
    """The record of one finding. Evidence paths are relative to ``folder``, where the records are saved."""
    fps, recorded_at = context.fps, context.recorded_at
    plate_fields = plate_summary(plate)
    frames = {"start": finding.start_frame, "trigger": finding.trigger_frame, "end": finding.end_frame}
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id(context.video_id, finding),
        "camera_id": context.camera_id,
        "video": {"video_id": context.video_id, "fps": fps, "width": context.width, "height": context.height, "recorded_at": recorded_at.isoformat()},
        "violation_type": finding.rule,
        "track": {"tracking_session": context.tracking_session, "track_id": finding.track_id, "class_name": finding.class_name},
        "time": {
            **{f"{name}_frame": frame for name, frame in frames.items()},
            **{f"{name}_time_s": round(frame / fps, 3) for name, frame in frames.items()},
            **{f"{name}_at": time_at(recorded_at, frame / fps) for name, frame in frames.items()},
        },
        "rule": {"name": finding.rule, "version": context.settings["rules"][finding.rule]["version"]},
        "why": finding.why,
        "signal_source": finding.evidence.get("signal_source"),
        "measurements": {k: v for k, v in finding.evidence.items() if k != "signal_source"},
        "evidence": {
            "clip": relative_to(clip.path, folder),
            "clip_frames": [clip.start, clip.end],
            "plate_crop": relative_to(plate_image, folder) if plate_image else None,
            "frames": [{"role": i.role, "frame_in_clip": i.frame_in_clip, "source_frame": i.source_frame, "image": relative_to(i.path, folder)} for i in images],
        },
        "license_plate": plate_fields,
        "needs_attention": attention_flags(finding, plate_fields),
        "settings": {
            "perception": context.settings["perception"], "rule": context.settings["rules"][finding.rule],
            "plates": context.settings["plates"], "evidence": context.settings["evidence"],
        },
        "config_files": context.config_files,
        "models": context.models,
        "review": {"status": PENDING, "reviewer": None, "decided_at": None, "corrected_plate": None, "notes": None},
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Everything wrong with a record, as sentences. An empty list means it is fit for review."""
    problems = [f"{name} is missing" for name in REQUIRED_FIELDS if name not in record]
    problems += [f"{name} has the wrong type" for name, kind in REQUIRED_FIELDS.items() if name in record and not isinstance(record[name], kind)]
    if problems:
        return problems
    video_id, times, plate, evidence = record["video"].get("video_id"), record["time"], record["license_plate"], record["evidence"]
    if record["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if not video_id or video_id not in record["event_id"] or record["event_id"] == str(record["track"].get("track_id")):
        problems.append("event_id must name its video, and differ from the track id")
    if not str(record["track"].get("tracking_session", "")).startswith(f"{video_id}/"):
        problems.append("the track must be scoped to a tracking session of this video")
    if not (isinstance(times.get("trigger_frame"), int) and times["trigger_frame"] >= 0 and times.get("trigger_time_s", -1) >= 0):
        problems.append("the trigger time is invalid")
    try:
        parse_recorded_at(times.get("trigger_at", ""))
    except (TypeError, ValueError):
        problems.append("the trigger's absolute time needs a time zone")
    if plate.get("status") not in PLATE_STATUSES:
        problems.append("the plate status is unknown")
    elif (plate.get("text") is None) != (plate["status"] not in (READ, UNCERTAIN)):
        problems.append("the plate text does not fit its status")
    if record["review"].get("status") != PENDING or record["review"].get("reviewer") is not None:
        problems.append("a record from the service is pending, with no reviewer")
    paths = [evidence.get("clip"), evidence.get("plate_crop"), *(f.get("image") for f in evidence.get("frames", []))]
    if any(p is not None and (Path(p).is_absolute() or ".." in Path(p).parts) for p in paths):
        problems.append("evidence paths must be relative, inside the folder of the records")
    return problems


def same_occurrence(a: dict[str, Any], b: dict[str, Any], window_seconds: float) -> bool:
    """True if two records describe one violation: the same id, or one vehicle shortly apart."""
    if a["event_id"] == b["event_id"]:
        return True
    if a["video"]["video_id"] != b["video"]["video_id"] or a["violation_type"] != b["violation_type"]:
        return False
    if abs(a["time"]["trigger_time_s"] - b["time"]["trigger_time_s"]) > window_seconds:
        return False
    plates = a["license_plate"], b["license_plate"]
    same_plate = all(p["status"] == READ for p in plates) and plates[0]["text"] == plates[1]["text"]  # An uncertain text may be wrong.
    return a["track"]["track_id"] == b["track"]["track_id"] or same_plate


def deduplicate_records(records: list[dict[str, Any]], window_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first record of each occurrence. Returns the records kept, and those dropped with what they repeat."""
    kept: list[dict[str, Any]] = []
    dropped = []
    for record in sorted(records, key=lambda r: r["time"]["trigger_time_s"]):
        original = next((k for k in kept if same_occurrence(k, record, window_seconds)), None)
        if original is None:
            kept.append(record)
        else:
            dropped.append({**record, "duplicate_of": original["event_id"]})
    return kept, dropped


def save_records(records: list[dict[str, Any]], path: Path) -> None:
    """Save records as JSON. They hold real plate numbers, so the file must stay out of git."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "events": records}, indent=1))
