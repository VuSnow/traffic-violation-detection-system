"""Tests for traffic_violation_detection.camera, on small videos of a textured background."""

import cv2
import numpy as np
import pytest
from fakes import write_video

from traffic_violation_detection.camera import estimate_camera_motion, static_mask

SIZE = (480, 360)


def texture(seed=0):
    noise = np.random.default_rng(seed).integers(0, 256, (SIZE[1] + 60, SIZE[0] + 60, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 1.2)


def view(background, dx, dy):
    # What a camera shifted by (dx, dy) sees.
    return background[30 + dy:30 + dy + SIZE[1], 30 + dx:30 + dx + SIZE[0]].copy()


def car(x1=10, y1=20, x2=110, y2=120):
    return {"class_name": "car", "bbox_xyxy": [x1, y1, x2, y2]}


def test_mask_blocks_moving_objects_with_a_margin():
    mask = static_mask((200, 300), [car()], margin=5)
    assert mask[50, 50] == 0 and mask[124, 50] == 0 and mask[126, 50] == 255


def test_mask_keeps_traffic_lights_usable():
    mask = static_mask((200, 300), [{"class_name": "traffic light", "bbox_xyxy": [10, 10, 40, 60]}])
    assert (mask == 255).all()


def test_motion_is_recovered_on_a_shifting_camera(tmp_path):
    background = texture()
    shifts = [(0, 0), (4, 2), (8, 4), (12, 6), (16, 8)]
    path = write_video(tmp_path / "v.mp4", [view(background, *s) for s in shifts])
    params = estimate_camera_motion(path, [[] for _ in shifts], reference_frame=2)
    # A point of frame k maps to the reference by adding the camera's offset relative to the reference frame.
    for k, (dx, dy) in enumerate(shifts):
        assert params[k][2] == pytest.approx(dx - 8, abs=1.0) and params[k][3] == pytest.approx(dy - 4, abs=1.0)
        assert params[k][0] == pytest.approx(1.0, abs=0.01) and params[k][1] == pytest.approx(0.0, abs=0.01)


def test_moving_object_does_not_pull_the_estimate_when_masked(tmp_path):
    background = texture()
    patch = texture(seed=1)[:200, :200]
    frames, detections = [], []
    for k in range(5):
        frame = view(background, 0, 0)
        x = 20 + 30 * k
        frame[100:300, x:x + 200] = patch
        frames.append(frame)
        detections.append([car(x, 100, x + 200, 300)])
    params = estimate_camera_motion(write_video(tmp_path / "v.mp4", frames), detections, reference_frame=2)
    assert np.abs(params[:, 2:]).max() < 1.0


def test_unmatched_frame_keeps_the_previous_transform_and_warns(tmp_path):
    background = texture()
    frames = [view(background, 0, 0), view(background, 6, 0), np.zeros((SIZE[1], SIZE[0], 3), np.uint8), view(background, 0, 0)]
    with pytest.warns(UserWarning, match="1 of 4 frames could not be matched"):
        params = estimate_camera_motion(write_video(tmp_path / "v.mp4", frames), [[] for _ in frames], reference_frame=0)
    assert params[2].tolist() == params[1].tolist()


def test_reference_frame_outside_the_video_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        estimate_camera_motion(tmp_path / "v.mp4", [[], []], reference_frame=450)
