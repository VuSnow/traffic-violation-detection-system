"""The service's core, without HTTP: submit a job, see its state, get its records and files, list the cameras.

The REST layer only translates between HTTP and these calls (spec, Section 8). Jobs run one at a time,
in order of arrival: ``run_next`` runs the oldest queued one, and a worker calls it in a loop.
"""

from __future__ import annotations

import fcntl
import json
import re
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

from tvd.camera import load_rule_defaults
from tvd.records import parse_recorded_at
from tvd.service.cameras import CameraCatalog
from tvd.service.intake import Refused, estimate_seconds, inspect_video, save_upload
from tvd.service.jobs import DONE, RUNNING, Job, JobStore
from tvd.service.runner import Pipeline
from tvd.service.settings import ServiceSettings
from tvd.service.storage import JobFolders

SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")


class NotFound(Exception):
    """No such job, or no such file in it."""


class NotReady(Exception):
    """The job is not done, so it has no records yet. The message is its state."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_text(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


class ReviewService:
    """Everything is checked when it starts: the camera files, the settings, and the models, which load once.

    A job that was running when the service last stopped is marked failed, and finished jobs older than the
    retention are deleted with their folders. Only one service may use a storage folder: a second one would
    run the same queue.
    """

    def __init__(self, settings: ServiceSettings, clock: Callable[[], datetime] = utc_now, pipeline: Pipeline | None = None) -> None:
        self.settings = settings
        self.clock = clock
        self.cameras = CameraCatalog(settings.files.cameras, load_rule_defaults(settings.files.rules), settings.jobs.seconds_per_video_second)
        self.folders = JobFolders(settings.storage.folder)
        self._lock = (self.folders.root / "service.lock").open("w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise RuntimeError(f"Another service is using {self.folders.root}") from error
        self.store = JobStore(self.folders.database)
        self.pipeline = pipeline if pipeline is not None else Pipeline(settings.files)
        self._admission = threading.Lock()  # Two uploads at once must not both fit in the same room in the queue.
        self.interrupted = self.store.interrupt(as_text(clock()))
        self.remove_old()

    def submit(self, upload: BinaryIO, filename: str, camera_id: str, recorded_at: str) -> Job:
        """Save, check and queue an upload. ``Refused`` says why it was not queued; then nothing of it is kept."""
        try:
            parse_recorded_at(recorded_at)
        except ValueError as error:
            raise Refused(str(error)) from error
        camera = self.cameras.get(camera_id)
        if camera is None:
            raise Refused(f"Unknown camera {camera_id!r}. Known cameras: {', '.join(self.cameras.ids)}")
        now = self.clock()
        job_id = self.folders.new_id(now)
        suffix = Path(filename).suffix.lower()
        video = "video" + (suffix if SUFFIX.fullmatch(suffix) else ".mp4")  # The name sent is never used as a path.
        try:
            save_upload(upload, self.folders.folder(job_id) / video, self.settings.storage.max_upload_mb * 2**20)
            facts = inspect_video(self.folders.folder(job_id) / video, camera)
            estimate = estimate_seconds(facts, camera, self.settings.jobs)
            with self._admission:
                waiting = self.queue_seconds()
                deadline = self.settings.jobs.deadline_minutes * 60
                if waiting + estimate > deadline:
                    raise Refused(
                        f"The queue is full: this video would be reviewed in about {(waiting + estimate) / 60:.0f} min, "
                        f"after the deadline of {deadline / 60:g} min. Send it again later.", 503,
                    )
                job = Job(job_id, camera_id, recorded_at, video, round(facts.seconds, 3), round(estimate, 1), as_text(now))
                self.store.add(job)
        except BaseException:
            self.folders.remove(job_id)
            raise
        return job

    def queue_seconds(self) -> float:
        """How long the jobs now waiting should take, from their estimates: for the running one, what is left of it."""
        now = self.clock()
        total = 0.0
        for job in self.store.waiting():
            if job.state == RUNNING and job.started_at:
                total += max(0.0, job.estimate_seconds - (now - datetime.fromisoformat(job.started_at)).total_seconds())
            else:
                total += job.estimate_seconds
        return total

    def status(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise NotFound(f"No job {job_id}")
        return job

    def records(self, job_id: str) -> dict[str, Any]:
        """The job's ``records.json``, once it is done."""
        job = self.status(job_id)
        if job.state != DONE:
            raise NotReady(job.state)
        return json.loads((self.folders.folder(job_id) / "records.json").read_text())

    def file(self, job_id: str, path: str) -> Path:
        """One file of a job's folder, such as ``evidence/<event id>_entry.jpg``."""
        self.status(job_id)
        found = self.folders.file(job_id, path)
        if found is None:
            raise NotFound(f"No file {path} in job {job_id}")
        return found

    def describe_cameras(self) -> list[dict[str, Any]]:
        return self.cameras.describe()

    def run_next(self) -> Job | None:
        """Run the oldest queued job to its end, and return it; None if no job is queued. A failure is kept with its reason."""
        job = self.store.next_queued()
        if job is None:
            return None
        self.store.start(job.job_id, as_text(self.clock()))
        try:
            camera = self.cameras.get(job.camera_id)
            if camera is None:
                raise RuntimeError(f"Camera {job.camera_id} is no longer configured")

            def report(stage: str, done: int, total: int) -> None:
                self.store.progress(job.job_id, stage, done, total)

            self.pipeline.run(job, camera, self.folders.folder(job.job_id), report)
        except Exception as error:
            self.store.fail(job.job_id, f"{type(error).__name__}: {error}", as_text(self.clock()))
        else:
            self.store.finish(job.job_id, as_text(self.clock()))
        return self.store.get(job.job_id)

    def remove_old(self) -> list[str]:
        """Delete the finished jobs older than the retention, with their folders. Waiting jobs are never deleted."""
        cutoff = as_text(self.clock() - timedelta(days=self.settings.storage.keep_days))
        old = [job.job_id for job in self.store.finished_before(cutoff)]
        for job_id in old:
            self.folders.remove(job_id)
            self.store.delete(job_id)
        return old

    def close(self) -> None:
        self.pipeline.close()
        self._lock.close()  # Closing the file releases the lock.
