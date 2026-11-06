import pytest
from conftest import ROOT
from pydantic import ValidationError

from tvd.config import ConfigError, PerceptionConfig, load_config

GOOD = """
detector:
  type: test-moving-car
  options: {step: 2}
tracker:
  type: test-same-id
track_classes: [car]
"""


def write(tmp_path, text: str):
    path = tmp_path / "perception.yaml"
    path.write_text(text)
    return path


def test_a_good_file_loads(tmp_path):
    config = load_config(write(tmp_path, GOOD), PerceptionConfig)
    assert config.detector.options == {"step": 2} and config.tracker.options == {} and config.track_classes == ["car"]


def test_options_left_empty_mean_no_options(tmp_path):
    config = load_config(write(tmp_path, GOOD.replace("type: test-same-id", "type: test-same-id\n  options:")), PerceptionConfig)
    assert config.tracker.options == {}


def test_the_committed_settings_load():
    config = load_config(ROOT / "configs" / "tvd" / "perception.yaml", PerceptionConfig)
    assert (config.detector.type, config.tracker.type, config.tracker.options) == ("yolo", "ocsort", {"use_byte": True})


def test_an_unknown_key_is_named(tmp_path):
    with pytest.raises(ConfigError, match="trackr: Extra inputs are not permitted"):
        load_config(write(tmp_path, GOOD + "trackr: ocsort\n"), PerceptionConfig)


def test_a_missing_key_is_named(tmp_path):
    with pytest.raises(ConfigError, match="tracker: Field required"):
        load_config(write(tmp_path, GOOD.replace("tracker:\n  type: test-same-id\n", "")), PerceptionConfig)


def test_a_wrong_type_is_named(tmp_path):
    with pytest.raises(ConfigError, match="track_classes: Input should be a valid list"):
        load_config(write(tmp_path, GOOD.replace("[car]", "car")), PerceptionConfig)


def test_an_empty_list_of_classes_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="track_classes: List should have at least 1 item"):
        load_config(write(tmp_path, GOOD.replace("[car]", "[]")), PerceptionConfig)


def test_every_problem_is_reported_at_once(tmp_path):
    text = GOOD.replace("tracker:\n  type: test-same-id\n", "") + "trackr: ocsort\n"
    with pytest.raises(ConfigError, match="tracker: Field required; trackr: Extra inputs are not permitted"):
        load_config(write(tmp_path, text), PerceptionConfig)


def test_a_file_that_is_not_yaml_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "detector: [unclosed"), PerceptionConfig)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml", PerceptionConfig)


def test_settings_cannot_change_after_loading(tmp_path):
    config = load_config(write(tmp_path, GOOD), PerceptionConfig)
    with pytest.raises(ValidationError):
        config.track_classes = ["truck"]
