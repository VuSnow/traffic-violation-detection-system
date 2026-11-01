"""Video reading utilities, validated in notebooks/01_video_reading.ipynb.

Conventions (verified in Notebook 01):

- ``frame_index`` is 0-based and counted by the reader itself; it is never taken
  from ``CAP_PROP_POS_FRAMES``, which points at the *next* frame after a read.
- ``timestamp_s`` is the decoder's presentation time of the frame in seconds
  (``CAP_PROP_POS_MSEC`` right after the frame is grabbed).
- A time maps to the frame *on screen* at that time: ``floor(t * fps)``.
  OpenCV's own ``CAP_PROP_POS_MSEC`` seek rounds differently, so seeking is
  always done by frame index.
- Images are BGR ``uint8`` arrays, exactly as OpenCV returns them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np


class VideoMetadata(TypedDict):
    video_id: str
    path: str
    fps: float
    reported_frame_count: int
    width: int
    height: int
    reported_duration_s: float | None


class FrameRecord(TypedDict):
    """One decoded frame, following the Frame Data contract of the specification."""

    video_id: str
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    image: np.ndarray


@contextmanager
def open_video(path: str | Path) -> Iterator[cv2.VideoCapture]:
    """Open a video for reading and always release it when the with-block ends.

    Raises FileNotFoundError if ``path`` is not a file, and RuntimeError if
    OpenCV cannot open it (``cv2.VideoCapture`` itself never raises).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found (or not a file): {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open the video: {path}")
        yield capture
    finally:
        capture.release()


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Metadata as reported by the file header (not verified by decoding)."""
    with open_video(path) as capture:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "video_id": Path(path).stem,
            "path": str(path),
            "fps": fps,
            "reported_frame_count": frame_count,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_duration_s": frame_count / fps if fps > 0 else None,
        }


def iter_frames(
    path: str | Path, start: int = 0, stop: int | None = None, step: int = 1
) -> Iterator[FrameRecord]:
    """Yield frames ``start, start + step, ...`` (``< stop``, or until the video ends).

    Skipped frames are only grabbed (decoded, not converted), which is cheaper
    than ``read()``. If the video ends before the frame count in its header,
    a warning is emitted instead of finishing silently (e.g. truncated files).
    """
    if start < 0 or step < 1 or (stop is not None and stop < start):
        raise ValueError(f"Invalid range: start={start}, stop={stop}, step={step}")
    path = Path(path)
    with open_video(path) as capture:
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if start > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            landed = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if landed != start:
                # OpenCV clamps seeks past the end to the last decodable position.
                warnings.warn(f"{path.name}: cannot seek to frame {start}; the video ends at frame {landed}")
                return
        index = start
        while stop is None or index < stop:
            if not capture.grab():
                break
            if (index - start) % step == 0:
                ok, image = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"{path.name}: could not decode frame {index}")
                yield {
                    "video_id": path.stem,
                    "frame_index": index,
                    "timestamp_s": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "image": image,
                }
            index += 1
        expected_end = reported if stop is None else min(stop, reported)
        if index < expected_end:
            warnings.warn(
                f"{path.name}: video ended at frame {index}, but the header reports {reported} frames "
                f"(file truncated or frame count inaccurate)"
            )


def read_frame_at(path: str | Path, frame_index: int) -> FrameRecord:
    """Random access to a single frame. Seeking is costly; prefer ``iter_frames`` for runs of frames."""
    # closing() finishes the generator, releasing its capture, right after the first frame.
    with closing(iter_frames(path, start=frame_index, stop=frame_index + 1)) as frames:
        record = next(frames, None)
    if record is None:
        raise IndexError(f"Frame {frame_index} is beyond the end of {Path(path).name}")
    return record


def frame_index_at_time(t_seconds: float, fps: float) -> int:
    """Index of the frame on screen at time ``t`` (constant frame rate assumed)."""
    if t_seconds < 0:
        raise ValueError(f"Negative time: {t_seconds}")
    # Float products can land just below an exact frame boundary (2.32 s * 12.5 FPS
    # = 28.999999999999996); the epsilon keeps such times on the frame that starts there.
    return math.floor(t_seconds * fps + 1e-9)


def read_frame_at_time(path: str | Path, t_seconds: float) -> FrameRecord:
    fps = read_video_metadata(path)["fps"]
    return read_frame_at(path, frame_index_at_time(t_seconds, fps))
