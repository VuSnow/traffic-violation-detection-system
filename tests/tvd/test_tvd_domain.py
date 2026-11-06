import numpy as np
import pytest

from tvd.domain import Box, Frame


def test_box_measures():
    box = Box(10, 20, 50, 80)
    assert (box.width, box.height, box.area, box.center) == (40, 60, 2400, (30.0, 50.0))


def test_a_box_with_reversed_corners_is_refused():
    with pytest.raises(ValueError, match="reversed"):
        Box(50, 20, 10, 80)


@pytest.mark.parametrize("other, expected", [(Box(0, 0, 10, 10), 1.0), (Box(5, 0, 15, 10), 1 / 3), (Box(20, 20, 30, 30), 0.0)])
def test_iou(other, expected):
    assert Box(0, 0, 10, 10).iou(other) == pytest.approx(expected)


def test_a_box_with_no_area_overlaps_nothing():
    assert Box(5, 5, 5, 5).iou(Box(5, 5, 5, 5)) == 0.0


def test_frames_are_never_equal_by_value():
    image = np.zeros((2, 2, 3), np.uint8)
    assert Frame(0, 0.0, image) != Frame(0, 0.0, image)
