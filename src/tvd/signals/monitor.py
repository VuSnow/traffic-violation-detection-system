"""Signal states over a video: read in every frame of the pass, then smoothed."""

from __future__ import annotations

from tvd.domain import Detection, Frame
from tvd.observers import FrameObserver
from tvd.signals.base import UNKNOWN, BaseSignalReader, combine_states, smooth_states


class SignalTimeline:
    """The smoothed state of each signal group in each analysed frame."""

    def __init__(self, states: dict[str, dict[int, str]]) -> None:
        self._states = states

    @property
    def groups(self) -> list[str]:
        return list(self._states)

    def get(self, group: str, frame_index: int, default: str | None = None) -> str | None:
        """The state of a group in a frame, or ``default`` for a frame that was not analysed."""
        return self._states.get(group, {}).get(frame_index, default)

    def state(self, group: str, frame_index: int) -> str:
        return self.get(group, frame_index, UNKNOWN)

    def as_dict(self) -> dict[str, dict[int, str]]:
        return {group: dict(states) for group, states in self._states.items()}


class SignalMonitor(FrameObserver):
    """Reads every group of signal heads in every frame. A group without heads is always ``unknown``.

    The states are smoothed only at the end, by ``timeline()``, because a sliding window needs the frames after each one.
    """

    def __init__(self, groups: dict[str, list[BaseSignalReader]], smoothing_frames: int = 7) -> None:
        self.groups = groups
        self.smoothing_frames = smoothing_frames
        self._frames: list[int] = []
        self._raw: dict[str, list[str]] = {group: [] for group in groups}

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self._frames.append(frame.index)
        for group, readers in self.groups.items():
            self._raw[group].append(combine_states([r.read(frame.image) for r in readers]) if readers else UNKNOWN)

    def timeline(self) -> SignalTimeline:
        return SignalTimeline({
            group: dict(zip(self._frames, smooth_states(raw, self.smoothing_frames))) for group, raw in self._raw.items()
        })
