"""Trackers. Importing this package registers every tracker in it."""

from tvd.tracking.base import TRACKERS, BaseTracker

__all__ = ["TRACKERS", "BaseTracker"]
