import importlib.util

import pytest
from conftest import ROOT

from tvd.camera import CameraConfig, load_rule_defaults
from tvd.config import ComponentConfig, ConfigError
from tvd.factory import build_detector, build_motion_estimator, build_plate_models, build_rules, build_tracker

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


CAMERA = {
    "camera_id": "gate-1",
    "frame_size": [640, 360],
    "rules": {
        "wrong_way": {"scene": {"vanishing_point": [320, -1000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}},
        "red_light": {
            "scene": {"stop_line": [[0, 200], [640, 200]], "lane_dividers": [100], "lane_direction_deg": -90, "traffic_direction": "away_from_camera",
                      "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}},
            "settings": {"pass_margin": 0.3},
        },
    },
}


def camera(**changes) -> CameraConfig:
    return CameraConfig.model_validate({**CAMERA, **changes})


def test_rules_are_built_with_the_camera_settings_over_the_defaults():
    wrong_way, red_light = build_rules(camera(), load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml"))
    assert (wrong_way.name, red_light.name, red_light.settings.pass_margin, red_light.settings.entry_offset) == ("wrong_way", "red_light", 0.3, 0.3)


def test_an_unknown_rule_is_named():
    with pytest.raises(ConfigError, match="rules.speeding: Unknown rule 'speeding'. Available: red_light, wrong_way"):
        build_rules(camera(rules={"speeding": {"scene": {}}}), {})


def test_a_misspelt_camera_setting_is_named():
    rules = {"red_light": {**CAMERA["rules"]["red_light"], "settings": {"pass_margn": 0.3}}}
    with pytest.raises(ConfigError, match="rules.red_light: settings.pass_margn: Extra inputs are not permitted"):
        build_rules(camera(rules=rules), load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml"))


def test_the_motion_estimator_is_built_by_its_name():
    assert build_motion_estimator(ComponentConfig(type="orb", options={"reference_frame": 5})).reference_frame == 5


def test_plate_models_are_built_by_their_name():
    with build_plate_models(ComponentConfig(type="test-plate", options={"text": "FAKE9999"})) as models:
        assert models.text == "FAKE9999"


def test_a_misspelt_plate_model_option_is_reported():
    with pytest.raises(ConfigError, match="onnx_worker: .*unexpected keyword argument 'ocr'"):
        build_plate_models(ComponentConfig(type="onnx_worker", options={"ocr": "cct-s-v2-global-model"}))
