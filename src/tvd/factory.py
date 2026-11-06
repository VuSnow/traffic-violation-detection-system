"""Builds components from their settings: the one place where names in a file become objects.

The pipeline receives a detector and a tracker and never builds them itself (the dependency inversion
principle), so changing either is a change of settings, not of code.
"""

from typing import TypeVar

from tvd.config import ComponentConfig, ConfigError
from tvd.detection import DETECTORS, BaseDetector
from tvd.registry import Registry
from tvd.tracking import TRACKERS, BaseTracker

T = TypeVar("T")


def build(registry: Registry[T], config: ComponentConfig) -> T:
    try:
        return registry.create(config.type, **config.options)
    except (ValueError, TypeError) as error:
        raise ConfigError(f"{config.type}: {error}") from error


def build_detector(config: ComponentConfig) -> BaseDetector:
    return build(DETECTORS, config)


def build_tracker(config: ComponentConfig) -> BaseTracker:
    return build(TRACKERS, config)
