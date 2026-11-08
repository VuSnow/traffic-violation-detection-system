"""Settings files and a camera for service tests: the test detector and tracker, fake plate models, and the committed sample video."""

import shutil
from pathlib import Path

from conftest import ROOT

from tvd.service.settings import load_service_settings

PERCEPTION = """
detector:
  type: test-moving-car
  options: {step: 0, step_y: 1}
tracker:
  type: test-same-id
track_classes: [car]
"""
PLATES = """
version: 1
models:
  type: test-plate
reading: {frames_per_track: 12, min_frame_gap: 4, vehicle_margin: 0.15, lowest_plate_center: 0.4, min_plate_width: 30, min_chars: 4, min_char_prob: 0.95, min_votes: 2, min_agreement: 0.67}
"""
# The test car comes down the image by 1 px per frame: through a red light, against the only allowed direction.
CAMERA = """
camera_id: test-camera
frame_size: [640, 360]
rules:
  red_light:
    scene:
      stop_line: [[0, 150], [640, 150]]
      lane_dividers: [0, 600]
      lane_direction_deg: 90
      traffic_direction: toward_camera
      signals: {reader: test-fixed-state, through: [{state: red}]}
    settings: {entry_offset: 0.0, pass_margin: 0.2}
  wrong_way:
    scene:
      vanishing_point: [320, -100000]
      regions: [{id: road, polygon: [[0, 0], [640, 0], [640, 360], [0, 360]], allowed_direction: away_from_camera, tolerance_deg: 60}]
"""


def service_files(tmp_path: Path, plates: str = PLATES) -> Path:
    """A service.yaml for tests, in tmp_path, with its own storage, cameras, perception and plate files."""
    (tmp_path / "cameras").mkdir()
    (tmp_path / "cameras" / "test-camera.yaml").write_text(CAMERA)
    (tmp_path / "perception.yaml").write_text(PERCEPTION)
    (tmp_path / "plates.yaml").write_text(plates)
    text = (ROOT / "configs" / "service.yaml").read_text()
    for old, new in (
        ("outputs/service", str(tmp_path / "storage")), ("configs/cameras", str(tmp_path / "cameras")),
        ("configs/tvd/perception.yaml", str(tmp_path / "perception.yaml")), ("configs/tvd/plates.yaml", str(tmp_path / "plates.yaml")),
        ("configs/tvd/rules.yaml", str(ROOT / "configs" / "tvd" / "rules.yaml")), ("configs/tvd/evidence.yaml", str(ROOT / "configs" / "tvd" / "evidence.yaml")),
    ):
        text = text.replace(old, new)
    (tmp_path / "service.yaml").write_text(text)
    return tmp_path / "service.yaml"


def settings_for(tmp_path: Path, plates: str = PLATES):
    return load_service_settings(service_files(tmp_path, plates))


def copy_video(source: Path, folder: Path, name: str = "video.mp4") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / name)
    return folder / name
