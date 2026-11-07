import json

from tvd.cli import main

SETTINGS = """
detector:
  type: test-moving-car
tracker:
  type: test-same-id
track_classes: [car]
"""


def test_track_writes_the_tracks_and_prints_a_summary(tmp_path, capsys, sample_video):
    settings = tmp_path / "perception.yaml"
    settings.write_text(SETTINGS)
    output = tmp_path / "out" / "tracks.json"
    assert main(["track", "--video", str(sample_video), "--config", str(settings), "--output", str(output)]) == 0
    assert "traffic_sample.mp4: 240 frames, 1 tracks, detector test-moving-car, tracker test-same-id" in capsys.readouterr().out
    data = json.loads(output.read_text())
    assert (data["frames"], data["tracks"]) == (240, 1)
    assert data["track_list"][0]["points"][0] == [0, 10.0, 50.0, 50.0, 80.0, 0.9]


def test_a_bad_settings_file_is_reported_without_a_traceback(tmp_path, capsys, sample_video):
    settings = tmp_path / "perception.yaml"
    settings.write_text(SETTINGS.replace("test-same-id", "deepsort"))
    assert main(["track", "--video", str(sample_video), "--config", str(settings), "--output", str(tmp_path / "t.json")]) == 2
    assert "error: deepsort: Unknown tracker 'deepsort'" in capsys.readouterr().err


CAMERA_FILE = """
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
"""
DOWN = SETTINGS.replace("type: test-moving-car", "type: test-moving-car\n  options: {step: 0, step_y: 1}")


def test_check_writes_the_findings_and_prints_a_summary(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE)
    (tmp_path / "perception.yaml").write_text(DOWN)
    output = tmp_path / "findings.json"
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(output), "--perception", str(tmp_path / "perception.yaml")]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "traffic_sample.mp4: 240 frames, 1 tracks, camera test-camera" in printed and "red_light: 1 candidates for review; candidate 1" in printed
    assert json.loads(output.read_text())["rules"]["red_light"]["findings"][0]["trigger_frame"] == 70


def test_check_refuses_a_video_of_another_camera(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE.replace("[640, 360]", "[1280, 720]"))
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(tmp_path / "f.json")]
    assert main(argv) == 2
    assert "error: Camera test-camera has 1280x720 frames, but traffic_sample.mp4 is 640x360" in capsys.readouterr().err
