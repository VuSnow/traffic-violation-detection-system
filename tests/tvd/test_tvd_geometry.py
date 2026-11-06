import numpy as np
import pytest

from tvd.geometry import point_in_polygon, polyline_y

LINE = [[0, 100], [100, 110], [300, 150]]
SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_polyline_height_between_its_points():
    assert polyline_y(LINE, 50) == pytest.approx(105) and polyline_y(LINE, 200) == pytest.approx(130)


def test_polyline_height_beyond_its_ends_is_the_nearest_end():
    assert polyline_y(LINE, np.array([-20.0, 400.0])).tolist() == [100.0, 150.0]


@pytest.mark.parametrize("point, inside", [((5, 5), True), ((10, 5), True), ((11, 5), False)])
def test_point_in_polygon(point, inside):
    assert point_in_polygon(SQUARE, point) is inside


def test_the_same_answers_as_the_notebooks():
    from traffic_violation_detection.red_light import line_y
    from traffic_violation_detection.scene import point_in_polygon as notebook_point_in_polygon

    xs = np.linspace(-50, 350, 41)
    assert polyline_y(LINE, xs).tolist() == line_y(LINE, xs).tolist()
    points = [(x, y) for x in range(-2, 13) for y in range(-2, 13)]
    assert [point_in_polygon(SQUARE, p) for p in points] == [notebook_point_in_polygon(SQUARE, p) for p in points]
