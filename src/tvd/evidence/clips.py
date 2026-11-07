"""Short clips of candidates, with original pixels: what a reviewer watches, and where plates are read.

A clip is cut in time only: nothing is drawn on it, and it is compressed close to losslessly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tvd.domain import Box
from tvd.evidence.settings import ClipSettings
from tvd.rules import Finding
from tvd.video import VideoReader, VideoWriter


def clip_range(trigger_frame: int, track_start: int, fps: float, first_frame: int, end_frame: int, settings: ClipSettings) -> tuple[int, int]:
    """The frames of a candidate's clip, from the first to one past the last, inside the analysed frames ``[first_frame, end_frame)``.

    It reaches back to the start of the vehicle's track, because a vehicle seen from behind is nearest the
    camera, with its plate largest, before it reaches the line; but no further than ``max_pre_seconds``.
    """
    if not first_frame <= trigger_frame < end_frame:
        raise ValueError(f"Trigger frame {trigger_frame} is outside the analysed frames [{first_frame}, {end_frame})")
    wanted_start = min(track_start, trigger_frame - round(settings.pre_seconds * fps))
    earliest = trigger_frame - round(settings.max_pre_seconds * fps)
    start = max(first_frame, earliest, wanted_start)
    end = min(end_frame, trigger_frame + round(settings.post_seconds * fps) + 1)
    return start, end


@dataclass(frozen=True)
class Clip:
    """Frames ``start`` to ``end`` (not included) of the source video, saved at ``path``.

    ``boxes`` are the vehicle's boxes by frame of the clip, which counts from 0: only frames where it was tracked have one.
    """

    path: Path
    start: int
    end: int
    fps: float
    width: int
    height: int
    trigger_frame: int
    boxes: dict[int, Box]

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    @property
    def trigger_in_clip(self) -> int:
        return self.trigger_frame - self.start

    def box_near(self, frame_in_clip: int, tolerance: int) -> Box | None:
        """The vehicle's box at a frame of the clip, or at the nearest frame within ``tolerance``; None if it was not tracked there."""
        nearest = min(self.boxes, key=lambda k: abs(k - frame_in_clip), default=None)
        return self.boxes[nearest] if nearest is not None and abs(nearest - frame_in_clip) <= tolerance else None


def cut_clips(
    reader: VideoReader, findings: Mapping[str, Finding], first_frame: int, end_frame: int, folder: str | Path, settings: ClipSettings,
) -> dict[str, Clip]:
    """Cut one clip per finding, named by its key, in a single pass over the video.

    Decoding is most of the work, so every clip is written while the frames go by once. If anything fails,
    no clip is left behind.
    """
    folder = Path(folder)
    ranges = {
        name: clip_range(f.trigger_frame, f.points[0].frame_index if f.points else f.start_frame, reader.fps, first_frame, end_frame, settings)
        for name, f in findings.items()
    }
    if not ranges:
        return {}
    writers = {name: VideoWriter(folder / f"{name}.mp4", reader.fps, settings.crf) for name in ranges}
    try:
        for frame in reader.frames(min(s for s, _ in ranges.values()), max(e for _, e in ranges.values())):
            for name, (start, end) in ranges.items():
                if start <= frame.index < end:
                    writers[name].write(frame.image)
                    if frame.index == end - 1:
                        writers[name].close()
        short = {name: writers[name].frames_written for name, (start, end) in ranges.items() if writers[name].frames_written != end - start}
        if short:
            raise RuntimeError(f"The video ended before these clips were complete (frames written): {short}")
    except BaseException:
        for name, writer in writers.items():
            writer.abort()
            writer.path.unlink(missing_ok=True)
        raise
    return {
        name: Clip(
            path=writers[name].path, start=start, end=end, fps=reader.fps, width=reader.width, height=reader.height,
            trigger_frame=findings[name].trigger_frame,
            boxes={p.frame_index - start: p.box for p in findings[name].points if start <= p.frame_index < end},
        )
        for name, (start, end) in ranges.items()
    }
