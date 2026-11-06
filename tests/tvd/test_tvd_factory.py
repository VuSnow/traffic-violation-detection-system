import importlib.util

import pytest

from tvd.config import ComponentConfig, ConfigError
from tvd.factory import build_detector, build_tracker

needs_boxmot = pytest.mark.skipif(importlib.util.find_spec("boxmot") is None, reason="Needs BoxMOT, as on the GPU server.")


def test_a_detector_is_built_by_its_name_with_its_options():
    assert build_detector(ComponentConfig(type="test-moving-car", options={"step": 3})).step == 3


def test_an_unknown_tracker_lists_the_available_ones():
    with pytest.raises(ConfigError, match="Unknown tracker 'deepsort'. Available: botsort, bytetrack, ocsort"):
        build_tracker(ComponentConfig(type="deepsort"))


def test_an_option_the_class_does_not_take_is_reported():
    with pytest.raises(ConfigError, match="test-moving-car: .*unexpected keyword argument 'speed'"):
        build_detector(ComponentConfig(type="test-moving-car", options={"speed": 3}))


@needs_boxmot
def test_a_misspelt_boxmot_option_is_reported():
    with pytest.raises(ConfigError, match="ocsort: .*use_bytes"):
        build_tracker(ComponentConfig(type="ocsort", options={"use_bytes": True}))
