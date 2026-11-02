"""The whole baseline in one call, from a video to records ready for a person's review. Validated in Notebook 08.

Each step keeps its own function and folder, and ``run_pipeline`` at the end joins them. A step that
cannot finish does not stop what can still be done: if plates cannot be read, the violations are
kept, and their records say that plate reading was not attempted.

The optional ``review`` argument stands in for a person. Without it, every record stays ``pending``
and nothing reaches the database.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2

from traffic_violation_detection.detection import OUTPUT_CRF, load_detector
from traffic_violation_detection.events import APPROVED, PENDING, REJECTED, SqliteViolationStore, confirmed_record, decide, generate_review_events, plate_caption, rejection_record
from traffic_violation_detection.plates import load_plate_models, read_plates_from_clips
from traffic_violation_detection.red_light import detect_red_light_video, load_red_light_scene
from traffic_violation_detection.scene import load_scene_config
from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer
from traffic_violation_detection.wrong_way import detect_wrong_way_video

CAPTION_SECONDS = 3.0
VIOLATION_STEPS = {"red_light": detect_red_light_video, "wrong_way": detect_wrong_way_video}
SCENE_LOADERS = {"red_light": load_red_light_scene, "wrong_way": load_scene_config}


class StepTimes:
    """How long each step took: ``with times("step"): ...``."""

    def __init__(self):
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, step: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[step] = round(self.seconds.get(step, 0.0) + time.perf_counter() - start, 2)


def scene_kind(scene_path: str | Path) -> str:
    """Which rule a scene configuration is for: a red-light scene has clips, a wrong-way scene has regions."""
    scene = json.loads(Path(scene_path).read_text())
    return "red_light" if "clips" in scene else "wrong_way"


def source_frames(scene_path: str | Path, kind: str, frame_count: int) -> list[int]:
    """The source frame behind each frame of a violation step's video, which holds only a red-light scene's clips."""
    if kind == "red_light":
        clips = json.loads(Path(scene_path).read_text())["clips"].values()
        return [k for clip in clips for k in range(clip["start"], clip["end"])]
    return list(range(frame_count))


def read_plates_if_possible(manifest: Path, folder: Path, plate_models: tuple | None) -> tuple[Path | None, str | None]:
    """Read the plates of a violation step's clips. A failure is returned, because the violations must survive it."""
    try:
        return read_plates_from_clips(manifest, folder, plate_models=plate_models if plate_models is not None else load_plate_models())["plates_file"], None
    except (RuntimeError, OSError) as error:
        return None, f"Plate reading failed: {error}"


def check_identities(violation_ids: set, clip_ids: set, plate_ids: set | None, record_ids: set, duplicate_ids: set) -> list[str]:
    """Every event must keep its id through every step. Returns what went wrong, as sentences."""
    problems = []
    if clip_ids != violation_ids:
        problems.append(f"clips differ from violations: {sorted(clip_ids ^ violation_ids)}")
    if plate_ids is not None and plate_ids != violation_ids:
        problems.append(f"plates differ from violations: {sorted(plate_ids ^ violation_ids)}")
    if (record_ids | duplicate_ids) != violation_ids:
        problems.append(f"records differ from violations: {sorted((record_ids | duplicate_ids) ^ violation_ids)}")
    return problems


def apply_review(records: list[dict], review: Callable[[dict], dict | None], decided_at: str) -> list[dict]:
    """Ask ``review`` about each record. It returns ``decide``'s arguments, or None to leave the record pending."""
    decided = []
    for record in records:
        answer = review(record)
        decided.append(record if answer is None else decide(record, decided_at=decided_at, **answer))
    return decided


def store_decisions(decided: list[dict], database_path: Path) -> tuple[int, list[dict]]:
    """Save approved events in a new database. Returns how many were stored, and the audit records of rejected events."""
    database_path.unlink(missing_ok=True)
    store = SqliteViolationStore(database_path)
    try:
        for record in decided:
            if record["review"]["status"] == APPROVED:
                store.save(confirmed_record(record))
        return store.count(), [rejection_record(r) for r in decided if r["review"]["status"] == REJECTED]
    finally:
        store.close()


def draw_caption(image, lines: list[str]) -> None:
    """Lines of text in a dark band at the bottom of a frame, in place."""
    scale = max(0.5, 0.8 * image.shape[0] / 1080)
    thickness = max(1, round(scale * 2.5))
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for line in lines]
    line_height = max(height for _, height in sizes) + 12
    top = image.shape[0] - line_height * len(lines) - 8
    cv2.rectangle(image, (0, top), (max(width for width, _ in sizes) + 16, image.shape[0]), (0, 0, 0), -1)
    for number, line in enumerate(lines):
        cv2.putText(image, line, (8, top + line_height * (number + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def captions_at(frame: int, records: list[dict], fps: float) -> list[str]:
    """The captions of the events that happened within the last few seconds, at a source frame."""
    window = round(CAPTION_SECONDS * fps)
    return [
        f"{r['violation_type']} at {r['time']['trigger_time_s']:.1f} s, plate: {plate_caption(r['license_plate'])} ({r['review']['status']})"
        for r in records
        if r["time"]["trigger_frame"] <= frame < r["time"]["trigger_frame"] + window
    ]


def write_final_video(annotated: Path, output_path: Path, frames: list[int], records: list[dict], fps: float) -> None:
    """A copy of a violation step's video with each event's plate result shown for a few seconds after it happens."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        shutil.copyfile(annotated, output_path)
        return
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for record, frame in zip(iter_frames(annotated), frames):
            lines = captions_at(frame, records, fps)
            if lines:
                draw_caption(record["image"], lines)
            write_frame(record["image"])


def run_pipeline(
    video_path: str | Path,
    output_folder_path: str | Path,
    scene_path: str | Path,
    review: Callable[[dict], dict | None] | None = None,
    model=None,
    plate_models: tuple | None = None,
) -> dict:
    """Process a video from start to finish, with the rule its scene configuration is for.

    ``output_folder_path`` receives ``<video name>_output.mp4`` and ``<video name>_summary.json``, and
    each step's outputs in ``violations/``, ``plates/`` and ``events/``. With a ``review``, it also
    receives ``<video name>_decisions.json``, ``<video name>_audit.json`` and ``violations.db``, and
    this earlier file is replaced.

    ``review(record)`` returns the keyword arguments of ``decide``, or None to leave a record pending.
    Only a real reviewer can confirm a violation: without ``review`` nothing is stored.
    """
    video_path, folder = Path(video_path), Path(output_folder_path)
    meta = read_video_metadata(video_path)
    kind = scene_kind(scene_path)
    SCENE_LOADERS[kind](scene_path, meta["width"], meta["height"])  # A wrong scene fails before the slow steps.
    video_id, fps = meta["video_id"], meta["fps"]
    times = StepTimes()

    with times("load the detector"):
        detector = model if model is not None else load_detector()
    with warnings.catch_warnings(record=True) as caught, times("detect, track and apply the rule"):
        warnings.simplefilter("always")
        violations = VIOLATION_STEPS[kind](video_path, folder / "violations", scene_path, model=detector)
    with times("read plates"):
        plates_file, plate_failure = read_plates_if_possible(violations["clips_manifest"], folder / "plates", plate_models)
    with times("build records and evidence"):
        events = generate_review_events(video_path, folder / "events", folder / "violations", plates_file)
    records, decided, stored, audit = events["records"], events["records"], 0, []
    with times("review and store"):
        if review is not None:
            decided = apply_review(records, review, datetime.now(timezone.utc).isoformat(timespec="seconds"))
            stored, audit = store_decisions(decided, folder / "violations.db")
            (folder / f"{video_id}_decisions.json").write_text(json.dumps(decided, indent=1))
            (folder / f"{video_id}_audit.json").write_text(json.dumps(audit, indent=1))
    output_video = output_video_path(video_path, folder)
    with times("write the final video"):
        frames = source_frames(scene_path, kind, sum(1 for _ in iter_frames(violations["output_video"])) if kind == "wrong_way" else 0)
        write_final_video(violations["output_video"], output_video, frames, decided, fps)

    violation_ids = {e["event_id"] for e in violations["events"]}
    clip_ids = {c["event_id"] for c in json.loads(Path(violations["clips_manifest"]).read_text())["clips"]}
    plate_ids = {e["event_id"] for e in json.loads(Path(plates_file).read_text())["events"]} if plates_file else None
    problems = check_identities(violation_ids, clip_ids, plate_ids, {r["event_id"] for r in records}, {r["event_id"] for r in events["duplicates"]})
    if problems:
        raise ValueError(f"Event identities changed between steps: {problems}")

    total = round(sum(times.seconds.values()), 2)
    outcomes = violations["crossings"] if kind == "red_light" else None
    summary = {
        "video": {k: meta[k] for k in ("video_id", "fps", "width", "height", "reported_frame_count", "reported_duration_s")},
        "rule": kind,
        "frames_analysed": len(frames),
        "seconds_of_video": round(len(frames) / fps, 2),
        "steps_seconds": times.seconds,
        "total_seconds": total,
        "times_real_time": round(len(frames) / fps / total, 2) if total else None,
        "outcomes": dict(Counter(r["verdict"] for clip in outcomes.values() for r in clip)) if outcomes is not None else dict(Counter(violations["track_statuses"].values())),
        "candidates": len(violations["events"]),
        "duplicates_dropped": len(events["duplicates"]),
        "plates": dict(Counter(r["license_plate"]["status"] for r in records)),
        "review": dict(Counter(r["review"]["status"] for r in decided)),
        "stored": stored,
        "failures": [plate_failure] if plate_failure else [],
        "warnings": sorted({str(w.message) for w in caught}),
    }
    summary_file = folder / f"{video_id}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=1))
    return {
        "output_video": output_video,
        "summary_file": summary_file,
        "summary": summary,
        "records": records,
        "decided": decided,
        "audit": audit,
        "records_file": events["records_file"],
        "database": folder / "violations.db" if review is not None else None,
        "pending": sum(r["review"]["status"] == PENDING for r in decided),
    }
