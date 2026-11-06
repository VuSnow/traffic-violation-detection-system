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
