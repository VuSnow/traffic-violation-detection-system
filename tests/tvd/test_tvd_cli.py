import importlib.util
import json
from pathlib import Path

import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_RLR, RED_LIGHT_SCENE, RLR_VIDEO, cut_losslessly, load_json
from service_data import service_files

from tvd.cli import main
from tvd.plates import PLATE_MODELS, BasePlateModels

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


def test_check_refuses_a_reference_frame_beyond_the_video_before_any_work(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE.replace("frame_size: [640, 360]", "frame_size: [640, 360]\nmotion: {type: orb, options: {reference_frame: 500}}"))
    (tmp_path / "perception.yaml").write_text(DOWN)
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(tmp_path / "f.json"), "--perception", str(tmp_path / "perception.yaml")]
    assert main(argv) == 2
    assert "error: The camera's reference frame 500 is beyond the video's 240 frames" in capsys.readouterr().err and not (tmp_path / "f.json").exists()


def test_check_refuses_a_video_of_another_camera(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE.replace("[640, 360]", "[1280, 720]"))
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(tmp_path / "f.json")]
    assert main(argv) == 2
    assert "error: Camera test-camera has 1280x720 frames, but traffic_sample.mp4 is 640x360" in capsys.readouterr().err


PLATES_FILE = """
version: 1
models:
  type: test-plate
reading: {frames_per_track: 12, min_frame_gap: 4, vehicle_margin: 0.15, lowest_plate_center: 0.4, min_plate_width: 30, min_chars: 4, min_char_prob: 0.95, min_votes: 2, min_agreement: 0.67}
"""


def review_args(tmp_path, video, recorded_at: str = "2026-11-07T08:00:00+07:00") -> list[str]:
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE)
    (tmp_path / "perception.yaml").write_text(DOWN)
    (tmp_path / "plates.yaml").write_text(PLATES_FILE)
    return [
        "review", "--video", str(video), "--camera", str(tmp_path / "camera.yaml"), "--recorded-at", recorded_at, "--output", str(tmp_path / "job"),
        "--perception", str(tmp_path / "perception.yaml"), "--plates", str(tmp_path / "plates.yaml"),
    ]


def test_review_writes_a_folder_for_review_and_prints_a_summary(tmp_path, capsys, sample_video):
    assert main(review_args(tmp_path, sample_video)) == 0
    printed = capsys.readouterr().out
    assert "red_light: 1 candidates for review" in printed and f"1 records for review in {tmp_path / 'job'} (duplicates dropped: 0); plates: unreadable 1" in printed
    record = json.loads((tmp_path / "job" / "records.json").read_text())["events"][0]
    assert record["time"]["trigger_at"] == "2026-11-07T08:00:02.917+07:00"  # Frame 70, at 24 frames per second.
    assert set(record["config_files"]) == {"perception", "camera", "rules", "plates", "evidence"} and record["settings"]["rule"]["pass_margin"] == 0.2
    assert record["models"]["tracker"] == {"type": "test-same-id"} and record["models"]["plates"] == {"type": "FakePlateModels"}


def test_review_needs_the_time_zone_of_the_first_frame_before_any_work(tmp_path, capsys, sample_video):
    assert main(review_args(tmp_path, sample_video, recorded_at="2026-11-07T08:00:00")) == 2
    assert "error: recorded_at needs a time zone" in capsys.readouterr().err and not (tmp_path / "job").exists()


def test_review_refuses_a_folder_in_use(tmp_path, capsys, sample_video):
    argv = review_args(tmp_path, sample_video)
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "records.json").write_text("{}")
    assert main(argv) == 2
    assert "error: The output folder already holds a review (records.json)" in capsys.readouterr().err


@PLATE_MODELS.register("test-cannot-start")
class CannotStart(BasePlateModels):
    def __init__(self) -> None:
        raise RuntimeError("Plate models: no such model")

    def detect(self, image):
        return []

    def recognize(self, plate):
        raise NotImplementedError


def test_review_keeps_the_records_when_the_plate_models_cannot_start(tmp_path, capsys, sample_video):
    argv = review_args(tmp_path, sample_video)
    (tmp_path / "plates.yaml").write_text(PLATES_FILE.replace("test-plate", "test-cannot-start"))
    assert main(argv) == 0
    assert "failure: plate reading: the plate models did not start: Plate models: no such model" in capsys.readouterr().out
    assert json.loads((tmp_path / "job" / "records.json").read_text())["events"][0]["license_plate"]["status"] == "not_attempted"


needs_gpu_rlr_and_notebook_08 = pytest.mark.skipif(
    any(importlib.util.find_spec(n) is None for n in ("ultralytics", "boxmot", "onnxruntime", "fast_plate_ocr", "open_image_models"))
    or not RLR_VIDEO.is_file() or not (ROOT / "weights" / "yolo26m.pt").is_file() or not NOTEBOOK_08_RLR.is_dir(),
    reason="Needs the models, the weights, data/raw/rlr_occasions.mp4 and the outputs of Notebook 08, as on the GPU server.",
)


def comparable(record: dict, frame_offset: int) -> dict:
    """What both versions of a record say about a case, with frames counted from the start of the clip."""
    return {
        "violation_type": record["violation_type"], "track_id": record["track"]["track_id"], "class_name": record["track"]["class_name"],
        "trigger_frame": record["time"]["trigger_frame"] - frame_offset, "rule": {k: record["rule"][k] for k in ("name", "version")},
        "why": record["why"], "signal_source": record["signal_source"], "measurements": record["measurements"],
        "license_plate": record["license_plate"], "needs_attention": record["needs_attention"], "review": record["review"],
        "evidence_frames": [(f["role"], f["frame_in_clip"]) for f in record["evidence"]["frames"]],
    }


@needs_gpu_rlr_and_notebook_08
def test_review_reproduces_notebook_08_on_clip_c_cut_into_its_own_file(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    clip = load_json(RED_LIGHT_SCENE)["clips"]["C"]
    video = cut_losslessly(RLR_VIDEO, clip["start"], clip["end"], tmp_path / "rlr_occasions-C.mkv")
    try:
        argv = ["review", "--video", str(video), "--camera", "configs/cameras/rlr_occasions-C.yaml", "--recorded-at", "2026-11-07T20:00:00+07:00", "--output", str(tmp_path / "job")]
        assert main(argv) == 0
    finally:
        video.unlink()  # Real footage is not left behind.
    ours = json.loads((tmp_path / "job" / "records.json").read_text())["events"]
    theirs = [r for r in load_json(NOTEBOOK_08_RLR / "events" / "rlr_occasions_review_events.json")["events"] if r["track"]["clip"] == "C"]
    same = [comparable(r, 0) for r in ours] == [comparable(r, clip["start"]) for r in theirs]  # Outside the assert: a failure never prints a plate.
    assert same and len(ours) == 2
    clips = NOTEBOOK_08_RLR / "violations" / "rlr_occasions_violation_clips"
    assert all((tmp_path / "job" / o["evidence"]["clip"]).read_bytes() == (clips / Path(t["evidence"]["clip"]).name).read_bytes() for o, t in zip(ours, theirs))


def test_serve_checks_everything_then_listens_where_the_settings_say(tmp_path, monkeypatch):
    import uvicorn
    from fastapi import FastAPI

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: started.update(app=app, host=host, port=port))
    assert main(["serve", "--config", str(service_files(tmp_path))]) == 0
    assert (started["host"], started["port"]) == ("127.0.0.1", 8040) and isinstance(started["app"], FastAPI)
    import json
    import logging

    from tvd.service.logs import LOGGER

    for handler in list(logging.getLogger(LOGGER).handlers):  # So that later tests are not logged in this test's folder.
        logging.getLogger(LOGGER).removeHandler(handler)
        handler.close()
    logging.getLogger(LOGGER).propagate = True
    started_line = json.loads((tmp_path / "storage" / "logs" / "service.log").read_text().splitlines()[-1])
    assert started_line["event"] == "service_started" and started_line["port"] == 8040 and started_line["models"]["tracker"] == {"type": "test-same-id"}


def test_serve_refuses_a_broken_camera_file_before_it_listens(tmp_path, capsys, monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: pytest.fail("the service must not start"))
    config = service_files(tmp_path)
    camera = tmp_path / "cameras" / "test-camera.yaml"
    camera.write_text(camera.read_text().replace("pass_margin", "pass_margn"))
    assert main(["serve", "--config", str(config)]) == 2
    assert "test-camera.yaml: rules.red_light:" in capsys.readouterr().err
