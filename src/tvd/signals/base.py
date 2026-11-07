"""What every signal reader offers, and how the readings of several heads become one state."""

from abc import ABC, abstractmethod

import numpy as np

from tvd.registry import Registry

RED, YELLOW, GREEN = "red", "yellow", "green"
UNKNOWN = "unknown"    # Nothing could be read.
CONFLICT = "conflict"  # Heads of one group disagree.


class BaseSignalReader(ABC):
    """Reads the state of one signal head in one frame."""

    @abstractmethod
    def read(self, image: np.ndarray) -> str:
        """``red``, ``yellow``, ``green``, or ``unknown`` when the head cannot be read."""


SIGNAL_READERS: Registry[BaseSignalReader] = Registry("signal reader")


def combine_states(states: list[str]) -> str:
    """The state of a group of heads: ``unknown`` if none is readable, ``conflict`` if the readable ones disagree."""
    known = {s for s in states if s != UNKNOWN}
    if not known:
        return UNKNOWN
    return known.pop() if len(known) == 1 else CONFLICT


def smooth_states(states: list[str], window: int = 7) -> list[str]:
    """A majority vote over a sliding window, to remove single-frame flicker. A tie goes to the name sorted last."""
    half = window // 2
    smoothed = []
    for i in range(len(states)):
        votes = states[max(0, i - half):i + half + 1]
        smoothed.append(max(set(votes), key=lambda s: (votes.count(s), s)))
    return smoothed
