"""Three images of each candidate, from its clip: before, at and after the moment the rule fired.

A vehicle hidden at its entry cannot be seen in those frames, and another vehicle may fill them: its
images are then the estimated entry, the moment it is first seen, and after that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from tvd.drawing import FrameOverlay, draw_vehicle_box
from tvd.evidence.clips import Clip
from tvd.evidence.settings import ImageSettings
from tvd.video import VideoReader


@dataclass(frozen=True)
class EvidenceImage:
    role: str
    frame_in_clip: int
    source_frame: int
    path: Path


def evidence_frames(clip: Clip, offset_frames: int) -> list[tuple[str, int]]:
    """The role and the frame of the clip of each image, in order."""
    last = clip.frame_count - 1
    entry = min(clip.trigger_in_clip, last)
    if clip.boxes and entry < min(clip.boxes):
        first_seen = min(clip.boxes)
        return [("estimated_entry", entry), ("first_seen", first_seen), ("after", min(last, first_seen + offset_frames))]
    return [("before", max(0, entry - offset_frames)), ("entry", entry), ("after", min(last, entry + offset_frames))]


def write_evidence_images(clip: Clip, folder: str | Path, name: str, settings: ImageSettings, overlay: FrameOverlay | None = None) -> list[EvidenceImage]:
    """Save the images as ``<name>_<role>.jpg``, with the vehicle's box where it was tracked, over what ``overlay`` draws.

    Two roles that fall on one frame, at the end of a clip, give one image: the first role's.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    roles = evidence_frames(clip, round(settings.offset_seconds * clip.fps))
    wanted: dict[int, str] = {}
    for role, frame in roles:
        wanted.setdefault(frame, role)
    saved: dict[str, EvidenceImage] = {}
    with VideoReader(clip.path) as video:
        for frame in video.frames(min(wanted), max(wanted) + 1):
            if frame.index not in wanted:
                continue
            image, role = frame.image, wanted[frame.index]
            if overlay is not None:
                overlay.draw(image, clip.start + frame.index)
            box = clip.box_near(frame.index, settings.box_tolerance_frames)
            if box is not None:
                draw_vehicle_box(image, box)
            path = folder / f"{name}_{role}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]):
                raise RuntimeError(f"Could not write {path}")
            saved[role] = EvidenceImage(role, frame.index, clip.start + frame.index, path)
    return [saved[role] for role, _ in roles if role in saved]
