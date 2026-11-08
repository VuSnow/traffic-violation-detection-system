"""After the pass (``check.py``): each candidate's clip, plate, images and record, in one folder for review.

The folder holds ``records.json``, ``summary.json``, ``clips/``, ``evidence/`` and ``plates/``. A failed plate
reading never loses what was found: the record is kept, and says that plate reading was not attempted.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tvd.check import CheckResult
from tvd.evidence import EvidenceSettings, cut_clips, write_evidence_images
from tvd.plates import BasePlateModels, PlateResult, ReadingSettings, read_vehicle_plate, save_plate_image
from tvd.records import RecordContext, build_record, deduplicate_records, event_id, save_records, validate_record
from tvd.video import VideoReader


@dataclass
class ReviewResult:
    folder: Path
    check: CheckResult
    records: list[dict[str, Any]]
    duplicates: list[dict[str, Any]]
    failures: list[str]
    seconds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Counts only: a summary never holds a plate number."""
        return {
            "video": self.check.video, "camera_id": self.check.camera_id, "frames": self.check.frames, "tracks": self.check.tracks,
            "verdicts": {name: r.verdict_counts() for name, r in self.check.results.items()},
            "records": len(self.records), "duplicates_dropped": len(self.duplicates),
            "plates": dict(Counter(r["license_plate"]["status"] for r in self.records)),
            "failures": self.failures, "seconds": {"check": self.check.seconds, **self.seconds},
        }


REVIEW_FILES = ("records.json", "summary.json", "clips", "evidence", "plates")


def prepare_folder(folder: Path) -> None:
    """A folder with no earlier review, whose files could show the wrong moment. Other files, such as the uploaded video, may be there."""
    if folder.exists() and not folder.is_dir():
        raise FileExistsError(f"The output must be a folder: {folder}")
    earlier = [name for name in REVIEW_FILES if (folder / name).exists()]
    if earlier:
        raise FileExistsError(f"The output folder already holds a review ({', '.join(earlier)}): {folder}")
    folder.mkdir(parents=True, exist_ok=True)


def read_plates(
    clips: dict, models: BasePlateModels | None, settings: ReadingSettings, folder: Path, failures: list[str], unavailable: str,
) -> dict[str, tuple[PlateResult, Path | None]]:
    """Each candidate's plate, from its clip. A candidate whose reading fails is left out, and the failure noted."""
    if models is None:
        failures.append(f"plate reading: {unavailable}")
        return {}
    plates = {}
    for name, clip in clips.items():
        try:
            with VideoReader(clip.path) as video:
                result = read_vehicle_plate(video, clip.boxes, models, settings)
            plates[name] = (result, save_plate_image(result, folder, name))
        except Exception as error:  # Any failure here must not lose the violation: its record says the plate was not read.
            failures.append(f"plate reading of {name}: {type(error).__name__}: {error}")
    return plates


def review_check(
    reader: VideoReader, check: CheckResult, context: RecordContext, plate_models: BasePlateModels | None,
    reading: ReadingSettings, evidence: EvidenceSettings, folder: str | Path, plates_unavailable: str = "no plate models",
) -> ReviewResult:
    """Write the evidence and the records of every finding of a pass over ``reader``'s video.

    Without ``plate_models``, every record says that plate reading was not attempted, and ``plates_unavailable`` says why.
    """
    folder = Path(folder)
    prepare_folder(folder)
    findings = sorted(((name, f) for name, r in check.results.items() for f in r.findings), key=lambda item: item[1].trigger_frame)
    by_id = {event_id(context.video_id, f): (name, f) for name, f in findings}
    seconds, failures = {}, []

    began = time.perf_counter()
    clips = cut_clips(reader, {k: f for k, (_, f) in by_id.items()}, check.first_frame, check.first_frame + check.frames, folder / "clips", evidence.clip)
    seconds["clips"] = round(time.perf_counter() - began, 1)

    began = time.perf_counter()
    plates = read_plates(clips, plate_models, reading, folder / "plates", failures, plates_unavailable)
    seconds["plates"] = round(time.perf_counter() - began, 1)

    began = time.perf_counter()
    records = []
    for key, (rule, finding) in by_id.items():
        images = write_evidence_images(clips[key], folder / "evidence", key, evidence.images, check.results[rule].overlay)
        plate, plate_image = plates.get(key, (None, None))
        records.append(build_record(context, finding, clips[key], images, plate, plate_image, folder))
    kept, dropped = deduplicate_records(records, evidence.records.duplicate_seconds)
    problems = {r["event_id"]: p for r in kept if (p := validate_record(r))}
    if problems:
        raise ValueError(f"Records not fit for review: {problems}")  # A broken record must never reach a reviewer.
    save_records(kept, folder / "records.json")
    seconds["images and records"] = round(time.perf_counter() - began, 1)

    result = ReviewResult(folder, check, kept, dropped, failures, seconds)
    (folder / "summary.json").write_text(json.dumps(result.summary(), indent=1))
    return result
