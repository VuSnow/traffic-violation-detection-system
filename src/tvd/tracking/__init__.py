"""Trackers. Importing this package registers every tracker in it."""

from tvd.tracking import boxmot_trackers  # noqa: F401  Registers "bytetrack", "ocsort" and "botsort".
from tvd.tracking.base import TRACKERS, BaseTracker

__all__ = ["TRACKERS", "BaseTracker"]
