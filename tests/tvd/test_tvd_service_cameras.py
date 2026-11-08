import shutil

import pytest
from conftest import ROOT

from tvd.camera import load_rule_defaults
from tvd.config import ConfigError
from tvd.service.cameras import CameraCatalog

CAMERAS = ROOT / "configs" / "cameras"
DEFAULTS = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
SPEEDS = {"fixed": 2.2, "orb": 7.0}


def copy_cameras(tmp_path):
    folder = tmp_path / "cameras"
    shutil.copytree(CAMERAS, folder)
    return folder


def test_every_camera_file_is_loaded():
    catalog = CameraCatalog(CAMERAS, DEFAULTS, SPEEDS)
    assert catalog.ids == ["rlr_occasions-C", "traffic_video_modified", "video-test"]
    camera = catalog.get("traffic_video_modified")
    assert camera.motion_type == "orb" and camera.version.startswith("sha256:") and catalog.get("nowhere") is None


def test_each_camera_is_described_for_the_backend():
    assert CameraCatalog(CAMERAS, DEFAULTS, SPEEDS).describe()[0] == {
        "camera_id": "rlr_occasions-C", "frame_size": [1280, 720], "motion": "fixed", "moving": False, "rules": ["red_light"],
    }


def test_a_file_named_after_another_camera_is_refused(tmp_path):
    folder = copy_cameras(tmp_path)
    (folder / "video-test.yaml").rename(folder / "junction.yaml")
    with pytest.raises(ConfigError, match="camera_id 'video-test' must be the file's name, 'junction'"):
        CameraCatalog(folder, DEFAULTS, SPEEDS)


def test_a_wrong_rule_setting_names_the_file_and_the_rule(tmp_path):
    folder = copy_cameras(tmp_path)
    path = folder / "traffic_video_modified.yaml"
    path.write_text(path.read_text().replace("pass_margin: 0.2", "pass_margn: 0.2"))
    with pytest.raises(ConfigError, match=r"traffic_video_modified.yaml: rules.red_light: .*pass_margn"):
        CameraCatalog(folder, DEFAULTS, SPEEDS)


def test_a_camera_whose_speed_was_never_measured_is_refused():
    with pytest.raises(ConfigError, match="no measured speed for a 'orb' camera"):
        CameraCatalog(CAMERAS, DEFAULTS, {"fixed": 2.2})


def test_a_folder_without_cameras_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="No camera file"):
        CameraCatalog(tmp_path, DEFAULTS, SPEEDS)
