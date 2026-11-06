import numpy as np
import pytest

from tvd.motion import MOTION_ESTIMATORS, CameraMotion
from tvd.video import VideoReader


def test_the_fixed_camera_is_registered():
    assert "fixed" in MOTION_ESTIMATORS.names()


def test_a_fixed_camera_leaves_points_unchanged(sample_video):
    with VideoReader(sample_video) as reader:
        motion = MOTION_ESTIMATORS.create("fixed").estimate(reader, {}, 0, 10)
    assert motion.is_fixed and motion.to_reference(5, [[12.5, 7.25]]).tolist() == [[12.5, 7.25]]


def test_a_moving_camera_maps_both_ways():
    motion = CameraMotion(np.array([[1.0, 0.0, 10.0, -5.0], [0.0, 1.0, 0.0, 0.0]]), first_frame=100)
    assert motion.to_reference(100, [[0, 0]]).tolist() == [[10.0, -5.0]]
    assert motion.to_reference(101, [[2, 0]])[0].tolist() == pytest.approx([0.0, 2.0])  # A quarter turn.
    assert motion.to_frame(100, motion.to_reference(100, [[3, 4]]))[0].tolist() == pytest.approx([3.0, 4.0])


def test_a_frame_outside_the_estimate_is_refused():
    with pytest.raises(IndexError, match="Frame 102 is outside"):
        CameraMotion(np.zeros((2, 4)), first_frame=100).params_at(102)
