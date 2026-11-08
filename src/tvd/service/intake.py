"""What an upload must be before it becomes a job. An upload refused here is never queued, and says why.

``Refused.status`` is the HTTP status that the API answers with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from tvd.check import check_frame_size
from tvd.config import ConfigError
from tvd.factory import build_motion_estimator
from tvd.service.cameras import Camera
from tvd.service.settings import JobSettings
from tvd.video import VideoReader

CHUNK_BYTES = 1 << 20


class Refused(Exception):
    def __init__(self, reason: str, status: int = 422) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def save_upload(source: BinaryIO, path: Path, max_bytes: int) -> int:
    """Copy an upload to ``path`` in pieces, and return its size. A larger one is refused while it arrives, and nothing of it is kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with path.open("wb") as out:
            while chunk := source.read(CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise Refused(f"The upload is larger than {max_bytes / 2**20:.0f} MB", 413)
                out.write(chunk)
        if written == 0:
            raise Refused("The upload is empty")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return written


@dataclass(frozen=True)
class VideoFacts:
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def seconds(self) -> float:
        return self.frame_count / self.fps


def inspect_video(path: Path, camera: Camera) -> VideoFacts:
    """The facts of an uploaded video, if it is a video of this camera that can be reviewed."""
    try:
        with VideoReader(path) as reader:
            check_frame_size(reader, camera.config)
            facts = VideoFacts(reader.frame_count, reader.fps, reader.width, reader.height)
    except (RuntimeError, ConfigError) as error:  # Not a video, or a video of another camera's frame size.
        raise Refused(str(error)) from error
    if facts.frame_count <= 0:
        raise Refused(f"{path.name} has no frames")
    try:
        build_motion_estimator(camera.config.motion).check_frames(facts.frame_count)
    except ValueError as error:
        raise Refused(str(error)) from error
    return facts


def estimate_seconds(facts: VideoFacts, camera: Camera, jobs: JobSettings) -> float:
    """How long the job should take. A video too long to be done within the deadline, even with no queue, is refused."""
    longest = jobs.longest_video_seconds(camera.motion_type)
    if facts.seconds > longest:
        raise Refused(
            f"The video lasts {facts.seconds / 60:.1f} min, and a video of a {camera.motion_type} camera can last "
            f"{longest / 60:.1f} min at most, to be reviewed within {jobs.deadline_minutes:g} min"
        )
    return jobs.work_seconds(camera.motion_type, facts.seconds)
