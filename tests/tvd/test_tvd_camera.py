import pytest
from conftest import ROOT
from notebook_data import load_json, red_light_section, wrong_way_scene

from tvd.camera import RuleSection, effective_settings, load_camera, load_rule_defaults
from tvd.config import ConfigError
from tvd.rules.red_light import RedLightScene
from tvd.rules.wrong_way import WrongWayScene

CAMERAS = ROOT / "configs" / "cameras"
MINIMAL = """
camera_id: gate-1
frame_size: [640, 360]
rules:
  wrong_way:
    scene: {vanishing_point: [0, 0], regions: []}
"""


def write(tmp_path, text: str):
    path = tmp_path / "camera.yaml"
    path.write_text(text)
    return path


def test_the_committed_camera_files_load():
    red, wrong = load_camera(CAMERAS / "traffic_video_modified.yaml"), load_camera(CAMERAS / "video-test.yaml")
    assert (red.frame_size, red.motion.type, list(red.rules)) == ((1920, 1012), "orb", ["red_light"])
    assert red.rules["red_light"].settings == {"entry_offset": 0.0, "pass_margin": 0.2}
    assert (wrong.motion.options["reference_frame"], list(wrong.rules)) == (450, ["wrong_way"])


def test_the_committed_camera_files_say_what_the_notebook_scenes_say():
    scenes = ROOT / "configs" / "scenes"
    wrong, red = load_camera(CAMERAS / "video-test.yaml"), load_camera(CAMERAS / "traffic_video_modified.yaml")
    old_wrong, old_red = load_json(scenes / "video-test.json"), load_json(scenes / "traffic_video_modified.json")
    assert WrongWayScene.model_validate(wrong.rules["wrong_way"].scene) == WrongWayScene.model_validate(wrong_way_scene(old_wrong))
    assert wrong.motion.options["reference_frame"] == old_wrong["coordinates"]["reference_frame_index"]
    section = red_light_section(old_red, "main")
    assert RedLightScene.model_validate(red.rules["red_light"].scene) == RedLightScene.model_validate(section["scene"])
    assert red.rules["red_light"].settings == section["settings"]
    assert (red.motion.options["reference_frame"], red.motion.options["excluded_areas"]) == (old_red["camera"]["reference_frame"], old_red["camera"]["excluded_areas"])


def test_a_camera_without_motion_is_fixed(tmp_path):
    camera = load_camera(write(tmp_path, MINIMAL))
    assert camera.motion.type == "fixed" and camera.rules["wrong_way"].settings == {}


def test_settings_left_empty_mean_none(tmp_path):
    assert load_camera(write(tmp_path, MINIMAL + "    settings:\n")).rules["wrong_way"].settings == {}


def test_a_camera_needs_a_rule(tmp_path):
    with pytest.raises(ConfigError, match="rules: Dictionary should have at least 1 item"):
        load_camera(write(tmp_path, MINIMAL.split("rules:")[0] + "rules: {}\n"))


def test_an_unknown_key_is_named(tmp_path):
    with pytest.raises(ConfigError, match="rules.wrong_way.setings: Extra inputs are not permitted"):
        load_camera(write(tmp_path, MINIMAL + "    setings: {window_seconds: 3}\n"))


def test_a_frame_size_needs_two_numbers(tmp_path):
    with pytest.raises(ConfigError, match="frame_size"):
        load_camera(write(tmp_path, MINIMAL.replace("[640, 360]", "[640]")))


def test_a_camera_changes_only_the_settings_it_names():
    defaults = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
    settings = effective_settings(defaults, RuleSection(scene={}, settings={"window_seconds": 3.0}), "wrong_way")
    assert settings == {**defaults["wrong_way"], "window_seconds": 3.0} and defaults["wrong_way"]["window_seconds"] == 2.0
