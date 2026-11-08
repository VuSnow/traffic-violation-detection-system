import numpy as np
import pytest

from tvd.domain import Box, Detection
from tvd.motion import MOTION_ESTIMATORS, CameraMotion
from tvd.motion.orb import OrbMotionEstimator
from tvd.video import VideoReader


def detections_at(index: int) -> list[Detection]:
    return [Detection(Box(100 + index, 100, 200 + index, 180), 2, "car", 0.9), Detection(Box(300, 20, 320, 60), 9, "traffic light", 0.8)]


def test_the_estimators_are_registered():
    assert {"fixed", "orb"} <= set(MOTION_ESTIMATORS.names())


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


def test_the_reference_frame_must_be_analysed(sample_video):
    with VideoReader(sample_video) as reader, pytest.raises(ValueError, match="Reference frame 50 is outside"):
        OrbMotionEstimator(reference_frame=50).estimate(reader, {}, 0, 10)


def test_a_video_too_short_for_the_reference_frame_is_refused_before_any_work():
    with pytest.raises(ValueError, match="The camera's reference frame 240 is beyond the video's 240 frames"):
        OrbMotionEstimator(reference_frame=240).check_frames(240)
    OrbMotionEstimator(reference_frame=240).check_frames(241)
    MOTION_ESTIMATORS.create("fixed").check_frames(1)


def test_the_mask_hides_moving_objects_and_drawn_areas_only():
    mask = OrbMotionEstimator(reference_frame=0, excluded_areas=[[0, 0, 5, 5]], mask_margin=2).static_mask((360, 640), detections_at(0))
    assert mask[0, 0] == 0 and mask[100, 98] == 0 and mask[100, 97] == 255  # Excluded area; a car's box plus its 2 px margin.
    assert mask[30, 310] == 255  # A traffic light is part of the scene.


def test_the_same_motion_as_the_notebooks(sample_video):
    from traffic_violation_detection.camera import estimate_camera_motion

    stop, reference = 40, 20
    notebook_detections = [[{"class_name": d.class_name, "bbox_xyxy": d.box.as_list()} for d in detections_at(k)] for k in range(stop)]
    expected = estimate_camera_motion(sample_video, notebook_detections, reference)
    with VideoReader(sample_video) as reader:
        motion = OrbMotionEstimator(reference_frame=reference).estimate(reader, {k: detections_at(k) for k in range(stop)}, 0, stop)
    assert np.array_equal(np.array([motion.params_at(k) for k in range(stop)]), expected)
