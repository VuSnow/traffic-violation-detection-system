"""Tests for traffic_violation_detection.scene, using the committed scene configuration."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from traffic_violation_detection.scene import (
    allowed_direction_unit,
    frame_to_reference,
    load_camera_motion,
    load_scene_config,
    point_in_polygon,
    reference_to_frame,
    region_by_id,
)

SCENE_PATH = Path(__file__).resolve().parents[1] / "configs" / "scenes" / "video-test.json"
W, H = 1920, 1080


@pytest.fixture()
def scene():
    return load_scene_config(SCENE_PATH, W, H)


def write_scene(tmp_path, scene) -> Path:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    return path


def test_committed_scene_loads(scene):
    assert scene["scene_id"] == "video-test"
    assert {"left_carriageway", "right_carriageway", "junction"} <= {r["id"] for r in scene["regions"]}


def test_wrong_video_size_is_rejected():
    with pytest.raises(ValueError, match="video is 1280x720"):
        load_scene_config(SCENE_PATH, 1280, 720)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.update(schema_version=99), "schema_version"),
        (lambda s: s["regions"][0].update(polygon=[[800, 300], [900, 300]]), "at least 3 points"),
        (lambda s: s["regions"][0].update(polygon=[[800, 300], [850, 300], [900, 300]]), "no area"),
        (lambda s: s["regions"][0].update(allowed_direction="sideways"), "unknown direction"),
        (lambda s: s["lines"][0].update(points=[[930, 250], [933, 1070]]), "always-visible"),
        (lambda s: s["signals"][0].update(governs_regions=["nowhere"]), "unknown regions"),
        (lambda s: s["signals"][0].update(roi_xyxy=[1195, 96, 1138, 68]), "empty ROI"),
    ],
)
def test_broken_scenes_are_rejected(tmp_path, scene, mutate, message):
    broken = copy.deepcopy(scene)
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        load_scene_config(write_scene(tmp_path, broken), W, H)


def test_regions_contain_expected_points(scene):
    right = region_by_id(scene, "right_carriageway")["polygon"]
    assert point_in_polygon(right, (1100, 400))
    assert not point_in_polygon(right, (800, 300))


def test_allowed_direction_points_along_the_road(scene):
    vp = scene["vanishing_point"]
    away = allowed_direction_unit(region_by_id(scene, "right_carriageway"), (1100, 400), vp)
    toward = allowed_direction_unit(region_by_id(scene, "left_carriageway"), (800, 300), vp)
    assert away[1] < 0, "away from the camera is up in the image"
    assert toward[1] > 0, "toward the camera is down in the image"
    assert np.linalg.norm(away) == pytest.approx(1)


def test_region_without_direction_has_no_unit_vector(scene):
    with pytest.raises(ValueError, match="no allowed direction"):
        allowed_direction_unit(region_by_id(scene, "junction"), (800, 700), scene["vanishing_point"])


def test_identity_transform_leaves_points_unchanged():
    points = [[10.0, 20.0], [1500.0, 900.0]]
    np.testing.assert_allclose(frame_to_reference((1, 0, 0, 0), points), points)


def test_frame_and_reference_mappings_round_trip():
    params = (0.999, 0.01, -59.5, 34.8)
    points = np.array([[100.0, 200.0], [1500.0, 900.0]])
    np.testing.assert_allclose(reference_to_frame(params, frame_to_reference(params, points)), points, atol=1e-6)


def test_load_camera_motion(tmp_path):
    path = tmp_path / "motion.json"
    path.write_text(json.dumps({"frames": [{"frame_index": i, "a": 1, "b": 0, "tx": i, "ty": 0} for i in range(3)]}))
    np.testing.assert_array_equal(load_camera_motion(path)[:, 2], [0, 1, 2])

    path.write_text(json.dumps({"frames": [{"frame_index": 1, "a": 1, "b": 0, "tx": 0, "ty": 0}]}))
    with pytest.raises(ValueError, match="one record per frame"):
        load_camera_motion(path)
