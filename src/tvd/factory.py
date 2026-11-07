"""Builds components from their settings: the one place where names in a file become objects.

The pipeline receives its components and never builds them itself (the dependency inversion
principle), so changing any of them is a change of settings, not of code.
"""

from typing import Any, TypeVar

from tvd.camera import CameraConfig, effective_settings
from tvd.config import ComponentConfig, ConfigError
from tvd.detection import DETECTORS, BaseDetector
from tvd.motion import MOTION_ESTIMATORS, BaseMotionEstimator
from tvd.plates import PLATE_MODELS, BasePlateModels
from tvd.registry import Registry
from tvd.rules import RULES, BaseRule
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


def build_motion_estimator(config: ComponentConfig) -> BaseMotionEstimator:
    return build(MOTION_ESTIMATORS, config)


def build_plate_models(config: ComponentConfig) -> BasePlateModels:
    return build(PLATE_MODELS, config)


def build_rules(camera: CameraConfig, defaults: dict[str, dict[str, Any]]) -> list[BaseRule]:
    """The camera's rules, each with its defaults and the camera's own settings. A problem names the rule."""
    rules = []
    for name, section in camera.rules.items():
        try:
            rules.append(RULES.create(name, scene=section.scene, settings=effective_settings(defaults, section, name)))
        except (ValueError, TypeError) as error:
            raise ConfigError(f"rules.{name}: {error}") from error
    return rules
