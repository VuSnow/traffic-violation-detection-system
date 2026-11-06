"""Reading video files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

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
