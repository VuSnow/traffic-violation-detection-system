"""One vehicle's plate, from the frames of a video where it was tracked: a candidate's clip, in the service."""

from __future__ import annotations

from pathlib import Path

import cv2

from tvd.domain import Box
from tvd.plates.base import BasePlateModels
from tvd.plates.reading import read_vehicle, select_frames
from tvd.plates.settings import ReadingSettings
from tvd.plates.voting import PlateResult, vote
from tvd.video import VideoReader


def read_vehicle_plate(video: VideoReader, boxes: dict[int, Box], models: BasePlateModels, settings: ReadingSettings) -> PlateResult:
    """Read the plate in the frames where the vehicle looks largest, and let them vote. ``boxes`` are the vehicle's, by frame of the video."""
    wanted = set(select_frames(boxes, settings.frames_per_track, settings.min_frame_gap))
    reads = []
    if wanted:
        for frame in video.frames(min(wanted), max(wanted) + 1):
            if frame.index in wanted:
                read = read_vehicle(frame.image, frame.index, boxes[frame.index], models, settings)
                if read is not None:
                    reads.append(read)
    return vote(reads, settings)


def save_plate_image(result: PlateResult, folder: str | Path, name: str) -> Path | None:
    """Save the image of the result's best read, without further compression, as ``<name>_f<frame>.png``. None if there is no read."""
    read = result.best_read()
    if read is None:
        return None
    path = Path(folder) / f"{name}_f{read.frame:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), read.image):
        raise RuntimeError(f"Could not write {path}")
    return path
