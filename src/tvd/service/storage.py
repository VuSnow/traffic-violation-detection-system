"""Each job's folder, ``<storage>/<job id>/``: the uploaded video, then the review's files next to it."""

from __future__ import annotations

import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path

JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


class JobFolders:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    @property
    def database(self) -> Path:
        return self.root / "jobs.sqlite"

    def new_id(self, now: datetime) -> str:
        """The time of the upload, and 6 random hex digits: ids sort by time and never repeat in practice."""
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def folder(self, job_id: str) -> Path:
        """A job's folder. An id that is not one, such as ``../x``, is refused, so it never leads outside the storage."""
        if not JOB_ID.match(job_id):
            raise ValueError(f"Not a job id: {job_id!r}")
        return self.root / job_id

    def file(self, job_id: str, relative: str) -> Path | None:
        """A file inside a job's folder, or None. A path that leads outside the folder, even through a link, is never given."""
        folder = self.folder(job_id).resolve()
        path = (folder / relative).resolve()
        return path if path.is_relative_to(folder) and path.is_file() else None

    def remove(self, job_id: str) -> None:
        shutil.rmtree(self.folder(job_id), ignore_errors=True)
