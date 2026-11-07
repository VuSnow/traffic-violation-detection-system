"""What every violation rule offers, and what it is given and gives back."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Any

from tvd.motion import CameraMotion
from tvd.observers import FrameObserver
from tvd.registry import Registry
from tvd.tracking.history import Track, TrackPoint


@dataclass(frozen=True)
class Observation:
    """What one pass over a video saw: what every rule judges. ``end_frame`` is one past the last frame analysed."""

    tracks: dict[int, Track]
    motion: CameraMotion
    fps: float
    width: int
    height: int
    first_frame: int
    end_frame: int


@dataclass(frozen=True)
class Finding:
    """A candidate violation, for a person to review. A rule never decides that a violation happened.

    ``start_frame`` to ``end_frame`` is the stretch of video that shows it; ``trigger_frame`` is the moment the rule fired.
    ``points`` are the vehicle in every frame of its track, in frame pixels: where clips and plate reading find it.
    """

    rule: str
    track_id: str
    class_name: str
    start_frame: int
    trigger_frame: int
    end_frame: int
    why: str
    evidence: dict[str, Any] = field(default_factory=dict)
    points: tuple[TrackPoint, ...] = ()

    def as_json(self) -> dict[str, Any]:
        """Everything but the points: they are many, and the evidence keeps the vehicle's boxes in its clip."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "points"}


@dataclass
class RuleResult:
    """A rule's findings, a record of how it judged each track, and the settings it used."""

    findings: list[Finding]
    records: list[dict[str, Any]]
    settings: dict[str, Any]

    def verdict_counts(self) -> dict[str, int]:
        return dict(Counter(r["verdict"] for r in self.records))


class BaseRule(ABC):
    """Judges the tracks of one pass over a video.

    A rule that needs to see every frame, as the red-light rule reads the signals, returns its observers
    from ``frame_observers``, and the pass feeds them. The pass never needs to know which rules exist.
    """

    name: str

    def frame_observers(self) -> list[FrameObserver]:
        return []

    @abstractmethod
    def evaluate(self, observation: Observation) -> RuleResult:
        """Judge every track of the observation."""


RULES: Registry[BaseRule] = Registry("rule")
