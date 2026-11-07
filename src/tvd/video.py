"""Reading and writing video files."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from tvd.domain import Frame


class VideoReader:
    """The frames of a video file, in order. Use it in a ``with`` block, so that the file is closed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"OpenCV could not open the video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            self.close()
            raise RuntimeError(f"The video reports no frame rate: {self.path}")

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
        """Frames ``start`` to ``stop`` (not included), decoded from the first frame, so that every index is exact."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while stop is None or index < stop:
            if not self._capture.grab():
                return
            if index >= start:
                ok, image = self._capture.retrieve()
                if not ok:
                    return
                yield Frame(index, index / self.fps, image)
            index += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def find_ffmpeg() -> str:
    """The system's ffmpeg, or else the one that the imageio-ffmpeg package brings."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Writing video needs ffmpeg: install it, or the imageio-ffmpeg package") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoWriter:
    """Writes BGR frames to an H.264 file with ffmpeg. Use it in a ``with`` block, or call ``close`` or ``abort``.

    The size comes from the first frame. The file gets its name only when ``close`` succeeds, so a failure
    never leaves a partial video behind. A lower ``crf`` is better quality: 14 is close to lossless.
    The ffmpeg command is the notebooks' own, so that the same frames give the same file.
    """

    def __init__(self, path: str | Path, fps: float, crf: int) -> None:
        self.path = Path(path)
        self.fps = fps
        self.crf = crf
        self.frames_written = 0
        self._partial = self.path.with_name(self.path.stem + ".partial.mp4")
        self._process: subprocess.Popen | None = None
        self._shape: tuple[int, ...] | None = None

    def _command(self, width: int, height: int) -> list[str]:
        return [
            find_ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
            # An odd side gets one black row or column: H.264 in yuv420p needs even sides.
            # Without accurate_rnd, the conversion from BGR shifts every pixel by a few levels.
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-sws_flags", "accurate_rnd",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self._partial),
        ]

    def write(self, image: np.ndarray) -> None:
        if self._process is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._shape = image.shape
            height, width = image.shape[:2]
            self._process = subprocess.Popen(self._command(width, height), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        elif image.shape != self._shape:
            # ffmpeg reads raw bytes: a frame of another size would garble the video without any error.
            raise ValueError(f"Frame of shape {image.shape} in a video of shape {self._shape}")
        try:
            self._process.stdin.write(image.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg stopped while writing {self.path.name}: {self._process.stderr.read().decode(errors='replace').strip()}") from error
        self.frames_written += 1

    def close(self) -> None:
        """Finish the file and give it its name."""
        if self._process is None:
            raise RuntimeError(f"No frames to write to {self.path.name}")
        _, errors = self._process.communicate()
        if self._process.returncode != 0:
            self._partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed on {self.path.name}: {errors.decode(errors='replace').strip()}")
        self._partial.replace(self.path)

    def abort(self) -> None:
        """Stop without a file."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._partial.unlink(missing_ok=True)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *rest: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
