"""Jobs and their states, in a small SQLite table, so that a restart can still report every job.

A job is ``queued``, ``running``, ``done`` or ``failed``, and moves only forward. A job that was running
when the service stopped is marked ``failed`` when the service starts again, with the reason.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
INTERRUPTED = "The service stopped while this job was running"


@dataclass(frozen=True)
class Job:
    """Times are ISO 8601 in UTC. ``estimate_seconds`` is how long the job should take, from the measured speed."""

    job_id: str
    camera_id: str
    recorded_at: str
    video_file: str
    video_seconds: float
    estimate_seconds: float
    created_at: str
    state: str = QUEUED
    stage: str | None = None
    frames_done: int = 0
    frames_total: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    failure: str | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


COLUMNS = [f.name for f in fields(Job)]
SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, recorded_at TEXT NOT NULL, video_file TEXT NOT NULL,
    video_seconds REAL NOT NULL, estimate_seconds REAL NOT NULL, created_at TEXT NOT NULL, state TEXT NOT NULL,
    stage TEXT, frames_done INTEGER NOT NULL, frames_total INTEGER NOT NULL, started_at TEXT, finished_at TEXT, failure TEXT
)"""


class JobStore:
    """One connection per call, so that the API's threads and the worker's thread can share the file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:  # Commits at the end, or rolls back on an error.
                yield db
        finally:
            db.close()

    def _select(self, where: str, parameters: tuple = ()) -> list[Job]:
        """Jobs in order of arrival. Two that arrived in the same second keep the order in which they were added."""
        with self._connect() as db:
            rows = db.execute(f"SELECT {', '.join(COLUMNS)} FROM jobs WHERE {where} ORDER BY created_at, rowid", parameters).fetchall()
        return [Job(*row) for row in rows]

    def _update(self, job_id: str, states: tuple[str, ...], **values: Any) -> None:
        assignments = ", ".join(f"{name} = ?" for name in values)
        marks = ", ".join("?" for _ in states)
        with self._connect() as db:
            changed = db.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ? AND state IN ({marks})", (*values.values(), job_id, *states)).rowcount
        if changed != 1:
            raise ValueError(f"Job {job_id} is not {' or '.join(states)}")

    def add(self, job: Job) -> None:
        with self._connect() as db:
            db.execute(f"INSERT INTO jobs ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})", tuple(asdict(job).values()))

    def get(self, job_id: str) -> Job | None:
        found = self._select("job_id = ?", (job_id,))
        return found[0] if found else None

    def next_queued(self) -> Job | None:
        """The oldest queued job: jobs run in order of arrival."""
        found = self._select("state = ?", (QUEUED,))
        return found[0] if found else None

    def waiting(self) -> list[Job]:
        """The running job and the queued ones, in order."""
        return self._select("state IN (?, ?)", (RUNNING, QUEUED))

    def start(self, job_id: str, now: str) -> None:
        self._update(job_id, (QUEUED,), state=RUNNING, started_at=now)

    def progress(self, job_id: str, stage: str, frames_done: int, frames_total: int) -> None:
        self._update(job_id, (RUNNING,), stage=stage, frames_done=frames_done, frames_total=frames_total)

    def finish(self, job_id: str, now: str) -> None:
        self._update(job_id, (RUNNING,), state=DONE, finished_at=now)

    def fail(self, job_id: str, reason: str, now: str) -> None:
        self._update(job_id, (QUEUED, RUNNING), state=FAILED, failure=reason, finished_at=now)

    def interrupt(self, now: str) -> list[str]:
        """Mark every running job failed: the service has just started, so none of them is running any more."""
        running = [job.job_id for job in self._select("state = ?", (RUNNING,))]
        for job_id in running:
            self.fail(job_id, INTERRUPTED, now)
        return running

    def finished_before(self, moment: str) -> list[Job]:
        return self._select("state IN (?, ?) AND finished_at < ?", (DONE, FAILED, moment))

    def delete(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
