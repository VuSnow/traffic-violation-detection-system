"""What is loaded once for every job, and how one job is done: Part 2's pass, then Part 3's review."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from tvd.camera import load_rule_defaults
from tvd.check import check_video
from tvd.config import PerceptionConfig, file_version, load_config
from tvd.evidence import load_evidence_settings
from tvd.factory import build_detector, build_motion_estimator, build_plate_models, build_rules, build_tracker
from tvd.observers import FrameCounter
from tvd.plates import BasePlateModels, load_plates_config
from tvd.records import RecordContext, parse_recorded_at
from tvd.review import ReviewResult, review_check
from tvd.service.cameras import Camera
from tvd.service.jobs import Job
from tvd.service.logs import log_event
from tvd.service.settings import SettingsFiles
from tvd.versions import models_used
from tvd.video import VideoReader

PROGRESS_EVERY = 25  # frames between two progress reports

Report = Callable[[str, int, int], None]
"""``report(stage, frames_done, frames_total)``: the stages are tracking, camera motion or rules, then evidence."""


class Pipeline:
    """The detector, the tracker and the plate models, loaded once when the service starts, with the settings of every job.

    Each job gets new rules and a new motion estimator, built from its camera: they keep what they saw of one video.
    """

    def __init__(self, files: SettingsFiles) -> None:
        self.perception = load_config(files.perception, PerceptionConfig)
        self.rule_defaults = load_rule_defaults(files.rules)
        self.plates = load_plates_config(files.plates)
        self.evidence = load_evidence_settings(files.evidence)
        self.config_files = {name: file_version(getattr(files, name)) for name in ("perception", "rules", "plates", "evidence")}
        self.detector = build_detector(self.perception.detector)
        self.tracker = build_tracker(self.perception.tracker)
        self._plate_models: BasePlateModels | None = None
        self.plates_unavailable = "no plate models"
        self._start_plate_models()

    def _start_plate_models(self) -> None:
        try:
            self._plate_models = build_plate_models(self.plates.models)
        except RuntimeError as error:  # The review goes on without plates, and its records say why.
            self._plate_models, self.plates_unavailable = None, f"the plate models did not start: {error}"
            log_event("plate_models_unavailable", logging.WARNING, reason=self.plates_unavailable)

    def plate_models(self) -> BasePlateModels | None:
        """The plate models, started again if they are gone, as after a call that did not answer in time."""
        if self._plate_models is None or not self._plate_models.is_alive():
            if self._plate_models is not None:
                self._plate_models.close()
                log_event("plate_models_restarted", logging.WARNING)
            self._start_plate_models()
        return self._plate_models

    def run(self, job: Job, camera: Camera, folder: Path, report: Report) -> ReviewResult:
        """Review the job's video, which is in ``folder``, and write the review there."""
        with VideoReader(folder / job.video_file) as reader:
            total = reader.frame_count
            after_tracking = "camera motion" if camera.motion_type != "fixed" else "rules"
            counter = FrameCounter(lambda done: report("tracking" if done < total else after_tracking, done, total), PROGRESS_EVERY, total)
            report("tracking", 0, total)
            check = check_video(
                reader, camera.config, self.detector, self.tracker, set(self.perception.track_classes),
                build_motion_estimator(camera.config.motion), build_rules(camera.config, self.rule_defaults), observers=[counter],
            )
            report("evidence", check.frames, total)
            settings = {
                "perception": self.perception.model_dump(), "rules": {name: r.settings for name, r in check.results.items()},
                "plates": self.plates.model_dump(), "evidence": self.evidence.model_dump(),
            }
            plate_models = self.plate_models()
            context = RecordContext(
                job.job_id, reader.fps, reader.width, reader.height, camera.camera_id, parse_recorded_at(job.recorded_at),
                f"{job.job_id}/{self.perception.tracker.type}", settings, {**self.config_files, "camera": camera.version},
                models_used(self.detector, self.perception.tracker, plate_models, self.plates_unavailable),
            )
            return review_check(reader, check, context, plate_models, self.plates.reading, self.evidence, folder, self.plates_unavailable)

    def close(self) -> None:
        if self._plate_models is not None:
            self._plate_models.close()
