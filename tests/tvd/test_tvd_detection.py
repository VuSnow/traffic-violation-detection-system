import numpy as np
import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Frame


def test_a_detector_must_implement_detect():
    class Incomplete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_the_test_double_finds_a_moving_car():
    car = DETECTORS.create("test-moving-car", step=2).detect(Frame(3, 0.1, np.zeros((1, 1, 3), np.uint8)))[0]
    assert car.class_name == "car" and car.box.x1 == 16
