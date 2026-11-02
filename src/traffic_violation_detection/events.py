"""Violation events, developed in Notebook 07: one record per candidate, and what review does to it.

A record joins what the earlier steps found about a case: the rule's reasons, the vehicle's track,
the times, the plate, and evidence images. Every record starts ``pending``, and only a reviewer's
decision changes that: nothing in this module confirms a violation by itself. The schema is
documented in ``docs/event_schema.md``.

Small functions come first. ``generate_review_events`` joins them, from the output folders of a
violation step and of plate reading to the file of records, their evidence, and a video to review.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from traffic_violation_detection.detection import OUTPUT_CRF
from traffic_violation_detection.plates import NO_PLATE, READ, UNCERTAIN, UNREADABLE
from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer

EVENT_SCHEMA_VERSION = 1

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
NOT_ATTEMPTED = "not_attempted"

PLATE_STATUSES = (READ, UNCERTAIN, UNREADABLE, NO_PLATE, NOT_ATTEMPTED)
PLATE_ATTENTION = {UNCERTAIN: "plate_uncertain", UNREADABLE: "plate_unreadable", NO_PLATE: "plate_not_found", NOT_ATTEMPTED: "plate_not_attempted"}
DUPLICATE_SECONDS = 5.0
EVIDENCE_OFFSET_SECONDS = 0.7
JPEG_QUALITY = 90

REQUIRED_FIELDS = {
    "schema_version": int,
    "event_id": str,
    "video": dict,
    "violation_type": str,
    "track": dict,
    "time": dict,
    "rule": dict,
    "why": str,
    "signal_source": (str, type(None)),
    "measurements": dict,
    "evidence": dict,
    "license_plate": dict,
    "needs_attention": list,
    "review": dict,
}


def plate_summary(result: dict | None) -> dict:
    """What a record keeps of a plate result. Without one, plate reading was not attempted."""
    if result is None:
        return {"status": NOT_ATTEMPTED, "text": None, "votes": 0, "usable_reads": 0, "agreement": 0.0, "mean_prob": 0.0}
    return {k: result[k] for k in ("status", "text", "votes", "usable_reads", "agreement", "mean_prob")}


def attention_flags(event: dict, plate: dict) -> list[str]:
    """What a reviewer should look at closely, because the record alone cannot settle it."""
    flags = []
    if event.get("signal_source") == "simulated_overlay":
        flags.append("simulated_signal")
    measurements = event.get("evidence", {})
    if measurements.get("entry_estimated"):
        flags.append("entry_estimated")
    elif measurements.get("entry_seen") is False:
        flags.append("entry_not_seen")
    if plate["status"] in PLATE_ATTENTION:
        flags.append(PLATE_ATTENTION[plate["status"]])
    return flags


def build_record(event: dict, video: dict, tracking_session: str, plate_result: dict | None = None) -> dict:
    """The shared record of one candidate, from a violation step's event and its plate result."""
    plate = plate_summary(plate_result)
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event["event_id"],
        "video": {k: video[k] for k in ("video_id", "fps", "width", "height")},
        "violation_type": event["violation_type"],
        "track": {"tracking_session": event.get("tracking_session", tracking_session), "clip": event.get("clip"), "track_id": event["track_id"], "class_name": event["class_name"]},
        "time": {
            "start_frame": event.get("start_frame"),
            "trigger_frame": event["trigger_frame"],
            "end_frame": event.get("end_frame"),
            "start_time_s": event.get("start_time_s"),
            "trigger_time_s": event["trigger_time_s"],
            "end_time_s": event.get("end_time_s"),
        },
        "rule": event["rule"],
        "why": event["why"],
        "signal_source": event.get("signal_source"),
        "measurements": {**event["evidence"], **({"region_id": event["region_id"]} if "region_id" in event else {})},
        "evidence": {"clip": None, "plate_crop": None, "frames": []},
        "license_plate": plate,
        "needs_attention": attention_flags(event, plate),
        "review": {"status": PENDING, "reviewer": None, "decided_at": None, "corrected_plate": None, "notes": None},
    }


def validate_record(record: dict) -> list[str]:
    """Everything wrong with a record, as sentences. An empty list means it is fit for review."""
    problems = [f"{field} is missing" for field in REQUIRED_FIELDS if field not in record]
    problems += [f"{field} has the wrong type" for field, kind in REQUIRED_FIELDS.items() if field in record and not isinstance(record[field], kind)]
    if problems:
        return problems
    track, times, plate, review = record["track"], record["time"], record["license_plate"], record["review"]
    video_id = record["video"].get("video_id")
    if record["event_id"] == str(track.get("track_id")):
        problems.append("event_id must differ from the track id")
    if video_id not in record["event_id"]:
        problems.append("event_id must name its video")
    if not str(track.get("tracking_session", "")).startswith(f"{video_id}/"):
        problems.append("the track must be scoped to a tracking session of this video")
    if not (isinstance(times.get("trigger_frame"), int) and times["trigger_frame"] >= 0 and times.get("trigger_time_s", -1) >= 0):
        problems.append("the trigger time is invalid")
    if plate.get("status") not in PLATE_STATUSES:
        problems.append("the plate status is unknown")
    elif (plate.get("text") is None) != (plate["status"] not in (READ, UNCERTAIN)):
        problems.append("the plate text does not fit its status")
    if review.get("status") not in (PENDING, APPROVED, REJECTED):
        problems.append("the review status is unknown")
    elif review["status"] == PENDING and review.get("reviewer") is not None:
        problems.append("a pending record cannot have a reviewer")
    elif review["status"] != PENDING and not review.get("reviewer"):
        problems.append("a decision needs a reviewer")
    return problems


def same_occurrence(a: dict, b: dict, window_seconds: float = DUPLICATE_SECONDS) -> bool:
    """True if two records describe one violation: the same id, or one vehicle shortly apart."""
    if a["event_id"] == b["event_id"]:
        return True
    if a["video"]["video_id"] != b["video"]["video_id"] or a["violation_type"] != b["violation_type"]:
        return False
    if abs(a["time"]["trigger_time_s"] - b["time"]["trigger_time_s"]) > window_seconds:
        return False
    same_track = (a["track"]["clip"], a["track"]["track_id"]) == (b["track"]["clip"], b["track"]["track_id"])
    plates = a["license_plate"], b["license_plate"]
    same_plate = all(p["status"] == READ for p in plates) and plates[0]["text"] == plates[1]["text"]
    return same_track or same_plate


def deduplicate_records(records: list[dict], window_seconds: float = DUPLICATE_SECONDS) -> tuple[list[dict], list[dict]]:
    """Keep the first record of each occurrence. Returns the records kept, and those dropped with what they repeat."""
    kept: list[dict] = []
    dropped = []
    for record in sorted(records, key=lambda r: r["time"]["trigger_time_s"]):
        original = next((k for k in kept if same_occurrence(k, record, window_seconds)), None)
        if original is None:
            kept.append(record)
        else:
            dropped.append({**record, "duplicate_of": original["event_id"]})
    return kept, dropped


def box_near(clip_entry: dict, frame_in_clip: int, tolerance: int = 3):
    """The vehicle's box at a frame of its clip or close to it, or None if it was not tracked then."""
    boxes = clip_entry["vehicle_boxes"]
    nearest = min(boxes, key=lambda b: abs(b["frame_in_clip"] - frame_in_clip), default=None)
    return nearest["box_xyxy"] if nearest and abs(nearest["frame_in_clip"] - frame_in_clip) <= tolerance else None


def evidence_frame_numbers(clip_entry: dict) -> list[tuple[str, int]]:
    """Which frames of a clip to keep as images: before the entry, the entry, and after it.

    A vehicle hidden at its entry cannot be seen in those frames, and another vehicle may fill them.
    Its images are therefore the estimated entry, the moment it is first seen, and after that.
    """
    offset = round(EVIDENCE_OFFSET_SECONDS * clip_entry["fps"])
    last = clip_entry["frame_count"] - 1
    entry = min(clip_entry["trigger_frame_in_clip"], last)
    tracked = [b["frame_in_clip"] for b in clip_entry["vehicle_boxes"]]
    if tracked and entry < min(tracked):
        first_seen = min(tracked)
        return [("estimated_entry", entry), ("first_seen", first_seen), ("after", min(last, first_seen + offset))]
    return [("before", max(0, entry - offset)), ("entry", entry), ("after", min(last, entry + offset))]


def write_evidence_frames(clip_path: str | Path, clip_entry: dict, folder: str | Path) -> list[dict]:
    """Save a clip's evidence frames as images, with the vehicle's box when it was tracked there."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = dict((number, role) for role, number in reversed(evidence_frame_numbers(clip_entry)))
    saved = {}
    for record in iter_frames(clip_path, min(wanted), max(wanted) + 1):
        number = record["frame_index"]
        if number not in wanted:
            continue
        image = record["image"]
        box = box_near(clip_entry, number)
        if box is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), max(2, image.shape[0] // 360))
        name = f"{clip_entry['event_id']}_{wanted[number]}.jpg"
        cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved[wanted[number]] = {"role": wanted[number], "frame_in_clip": number, "source_frame": clip_entry["source_start_frame"] + number, "image": f"evidence/{name}"}
    return [saved[role] for role, _ in evidence_frame_numbers(clip_entry) if role in saved]


def relative_to(path: str | Path, folder: str | Path) -> str:
    return os.path.relpath(path, folder)


def attach_evidence(record: dict, clip_path: str | Path, clip_entry: dict, plate_crop_path: str | Path | None, output_folder: str | Path) -> dict:
    """The record with references to its clip, plate crop and evidence images. Paths are relative to the folder of the records."""
    evidence = {
        "clip": relative_to(clip_path, output_folder),
        "plate_crop": relative_to(plate_crop_path, output_folder) if plate_crop_path else None,
        "frames": write_evidence_frames(clip_path, clip_entry, Path(output_folder) / "evidence"),
    }
    return {**record, "evidence": evidence}


def save_records(records: list[dict], path: str | Path) -> None:
    """Save reviewable records as JSON. The file holds real plate numbers, so it must stay out of git."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"schema_version": EVENT_SCHEMA_VERSION, "events": records}, indent=1))


def load_records(path: str | Path) -> list[dict]:
    saved = json.loads(Path(path).read_text())
    if saved.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {saved.get('schema_version')}")
    return saved["events"]


def decide(record: dict, decision: str, reviewer: str, decided_at: str, corrected_plate: str | None = None, notes: str | None = None) -> dict:
    """The record with a reviewer's decision. The record given stays as it was.

    This is the only way out of ``pending``. A decision cannot be replaced here, because changing
    a decision is a separate act that a real review system has to log.
    """
    if decision not in (APPROVED, REJECTED):
        raise ValueError(f"Unknown decision: {decision}")
    if record["review"]["status"] != PENDING:
        raise ValueError(f"{record['event_id']} was already decided: {record['review']['status']}")
    if not reviewer or not reviewer.strip():
        raise ValueError("A decision needs a reviewer")
    datetime.fromisoformat(decided_at)  # Raises on a timestamp that cannot be read.
    return {**record, "review": {"status": decision, "reviewer": reviewer, "decided_at": decided_at, "corrected_plate": corrected_plate, "notes": notes}}


def final_plate(record: dict) -> tuple[str | None, str | None]:
    """The plate to keep, and who gave it: the reviewer, else the OCR if it was sure. A mere suggestion is never kept."""
    corrected = record["review"]["corrected_plate"]
    if corrected:
        return corrected, "reviewer"
    plate = record["license_plate"]
    return (plate["text"], "ocr") if plate["status"] == READ else (None, None)


def confirmed_record(record: dict) -> dict:
    """What is persisted for an approved violation. Anything else has no way in."""
    review = record["review"]
    if review["status"] != APPROVED:
        raise ValueError(f"{record['event_id']} is {review['status']}, and only an approved event can be persisted")
    plate_text, plate_source = final_plate(record)
    return {
        "event_id": record["event_id"],
        "video_id": record["video"]["video_id"],
        "violation_type": record["violation_type"],
        "trigger_time_s": record["time"]["trigger_time_s"],
        "vehicle_class": record["track"]["class_name"],
        "plate_text": plate_text,
        "plate_source": plate_source,
        "signal_source": record["signal_source"],
        "rule_name": record["rule"]["name"],
        "rule_version": record["rule"]["version"],
        "evidence": json.dumps(record["evidence"]),
        "reviewer": review["reviewer"],
        "decided_at": review["decided_at"],
        "notes": review["notes"],
    }


def rejection_record(record: dict) -> dict:
    """What is kept of a rejected event: the decision for audit, with no plate and no images of a vehicle cleared of it."""
    review = record["review"]
    if review["status"] != REJECTED:
        raise ValueError(f"{record['event_id']} is {review['status']}, not rejected")
    return {
        "event_id": record["event_id"],
        "video_id": record["video"]["video_id"],
        "violation_type": record["violation_type"],
        "rule_name": record["rule"]["name"],
        "rule_version": record["rule"]["version"],
        "reviewer": review["reviewer"],
        "decided_at": review["decided_at"],
        "notes": review["notes"],
    }


class ViolationStore(Protocol):
    """The interface of a database of confirmed violations."""

    def save(self, record: dict) -> None:
        """Store the result of ``confirmed_record``. An event can only be stored once."""

    def get(self, event_id: str) -> dict | None:
        """One confirmed violation, or None."""

    def find_by_plate(self, plate_text: str) -> list[dict]:
        """The confirmed violations of a plate, oldest first."""

    def count(self) -> int:
        """How many confirmed violations are stored."""


class SqliteViolationStore:
    """A small implementation of ``ViolationStore``, to test the interface. It is not the application's database."""

    COLUMNS = (
        "event_id", "video_id", "violation_type", "trigger_time_s", "vehicle_class", "plate_text", "plate_source",
        "signal_source", "rule_name", "rule_version", "evidence", "reviewer", "decided_at", "notes",
    )

    def __init__(self, path: str | Path = ":memory:"):
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS violations ({', '.join(c + (' TEXT PRIMARY KEY' if c == 'event_id' else '') for c in self.COLUMNS)})")

    def save(self, record: dict) -> None:
        missing = [c for c in self.COLUMNS if c not in record]
        if missing or not record["reviewer"]:
            raise ValueError(f"Not a confirmed record: missing {missing or 'reviewer'}")
        try:
            with self.connection:
                self.connection.execute(f"INSERT INTO violations VALUES ({', '.join('?' * len(self.COLUMNS))})", [record[c] for c in self.COLUMNS])
        except sqlite3.IntegrityError as error:
            raise ValueError(f"{record['event_id']} is already stored") from error

    def get(self, event_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM violations WHERE event_id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    def find_by_plate(self, plate_text: str) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM violations WHERE plate_text = ? ORDER BY decided_at", (plate_text,))
        return [dict(row) for row in rows]

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM violations").fetchone()[0]

    def close(self) -> None:
        self.connection.close()


def plate_caption(plate: dict) -> str:
    """A plate result in the words of a review video. A question mark says it is only a suggestion."""
    if plate["status"] == READ:
        return plate["text"]
    if plate["status"] == UNCERTAIN:
        return f"{plate['text']}?"
    return {UNREADABLE: "unreadable", NO_PLATE: "not found", NOT_ATTEMPTED: "not attempted"}[plate["status"]]


def draw_review_frame(image: np.ndarray, record: dict, position: str, box) -> None:
    """Draw what a reviewer needs on a frame of an event's clip, in place. The clip itself stays clean."""
    scale = max(0.5, 0.7 * image.shape[0] / 1080)
    thickness = max(1, round(scale * 2.5))
    if box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), max(2, image.shape[0] // 360))
    text = f"{position}  {record['violation_type']}  plate: {plate_caption(record['license_plate'])}  {record['review']['status']}"
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (0, 0), (width + 16, height + 16), (0, 0, 0), -1)
    cv2.putText(image, text, (8, height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def write_review_video(records: list[dict], clips: dict[str, tuple[Path, dict]], output_path: str | Path) -> None:
    """One video of all events in turn, with each event's plate result and status drawn on it."""
    fps = next(iter(clips.values()))[1]["fps"]
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for number, record in enumerate(records, start=1):
            clip_path, entry = clips[record["event_id"]]
            for frame in iter_frames(clip_path):
                draw_review_frame(frame["image"], record, f"{number}/{len(records)}", box_near(entry, frame["frame_index"], tolerance=0))
                write_frame(frame["image"])


def read_violation_outputs(video_id: str, violation_folder: Path) -> tuple[list[dict], str, dict]:
    """A violation step's events, its tracking session, and its clips by event id."""
    saved = json.loads((violation_folder / f"{video_id}_events.json").read_text())
    clips_folder = violation_folder / f"{video_id}_violation_clips"
    manifest = json.loads((clips_folder / "manifest.json").read_text())
    clips = {entry["event_id"]: (clips_folder / entry["clip_file"], entry) for entry in manifest["clips"]}
    return saved.get("events", saved.get("candidates", [])), saved["tracking_session"], clips


def read_plate_outputs(plates_file: Path | None) -> dict[str, dict]:
    """Plate results by event id, each with the path of its crop."""
    if plates_file is None:
        return {}
    saved = json.loads(Path(plates_file).read_text())
    folder = Path(plates_file).parent / "plate_crops"
    return {e["event_id"]: {"result": e["license_plate"], "crop": folder / e["plate_crop"] if e["plate_crop"] else None} for e in saved["events"]}


def generate_review_events(video_path: str | Path, output_folder_path: str | Path, violation_folder: str | Path, plates_file: str | Path | None = None) -> dict:
    """Turn the results of the earlier steps into records ready for a person's review.

    ``violation_folder`` is the output folder of a violation step, and ``plates_file`` the ``plates.json``
    of ``read_plates_from_clips``, if plates were read. In ``output_folder_path`` this saves:

    - ``<video name>_review_events.json``: one validated record per occurrence, all ``pending``;
    - ``evidence/``: three images per event;
    - ``<video name>_output.mp4``: all events in turn, for review. With no events there is no video.

    Earlier files of these names are replaced, and so is the whole of ``evidence/``. Returns these paths
    with the records. A record that fails validation raises an error, because a broken record must
    not reach a reviewer.
    """
    video_path, folder = Path(video_path), Path(output_folder_path)
    video = read_video_metadata(video_path)
    events, tracking_session, clips = read_violation_outputs(video["video_id"], Path(violation_folder))
    plates = read_plate_outputs(Path(plates_file) if plates_file else None)
    shutil.rmtree(folder / "evidence", ignore_errors=True)  # An earlier run's images could show the wrong moment.

    records = []
    for event in events:
        clip_path, entry = clips[event["event_id"]]
        plate = plates.get(event["event_id"], {})
        record = build_record(event, video, tracking_session, plate.get("result"))
        records.append(attach_evidence(record, clip_path, entry, plate.get("crop"), folder))
    kept, dropped = deduplicate_records(records)
    problems = {r["event_id"]: validate_record(r) for r in kept}
    problems = {event_id: found for event_id, found in problems.items() if found}
    if problems:
        raise ValueError(f"Records not fit for review: {problems}")

    records_file = folder / f"{video['video_id']}_review_events.json"
    save_records(kept, records_file)
    output_video = None
    if kept:
        output_video = output_video_path(video_path, folder)
        write_review_video(kept, clips, output_video)
    return {"records_file": records_file, "output_video": output_video, "records": kept, "duplicates": dropped}
