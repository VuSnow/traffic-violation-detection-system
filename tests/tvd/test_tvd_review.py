import json

import cv2
import numpy as np
import pytest
from conftest import ROOT, FakePlateModels

from tvd.camera import CameraConfig, load_rule_defaults
from tvd.check import check_video
from tvd.config import ComponentConfig
from tvd.evidence import load_evidence_settings
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
from tvd.plates import load_plates_config
from tvd.records import RecordContext, parse_recorded_at, validate_record
from tvd.review import review_check
from tvd.video import VideoReader

DEFAULTS = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
EVIDENCE = load_evidence_settings(ROOT / "configs" / "tvd" / "evidence.yaml")
# As in test_tvd_check.py: the test car comes down the image by 1 px per frame, through a red light, against the only allowed direction.
CAMERA = {
    "camera_id": "test-camera",
    "frame_size": [640, 360],
    "rules": {
        "red_light": {
            "scene": {"stop_line": [[0, 150], [640, 150]], "lane_dividers": [0, 600], "lane_direction_deg": 90, "traffic_direction": "toward_camera",
                      "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}},
            "settings": {"entry_offset": 0.0, "pass_margin": 0.2},
        },
        "wrong_way": {"scene": {"vanishing_point": [320, -100000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360], [0, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}},
    },
}


class BrokenPlateModels(FakePlateModels):
    def detect(self, image):
        raise RuntimeError("The plate models' process stopped")


def review(video, folder, plate_models=None, camera: dict = CAMERA):
    config = CameraConfig.model_validate(camera)
    with VideoReader(video) as reader:
        check = check_video(
            reader, config, build_detector(ComponentConfig(type="test-moving-car", options={"step": 0, "step_y": 1})),
            build_tracker(ComponentConfig(type="test-same-id")), {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS),
        )
        settings = {"perception": {"tracker": "test-same-id"}, "rules": {n: r.settings for n, r in check.results.items()}, "plates": PLATES.model_dump(), "evidence": EVIDENCE.model_dump()}
        context = RecordContext("traffic_sample", reader.fps, reader.width, reader.height, "test-camera", parse_recorded_at("2026-11-07T08:00:00+07:00"),
                                "traffic_sample/test-same-id", settings, {"camera": "sha256:000000000000"})
        return review_check(reader, check, context, plate_models, PLATES.reading, EVIDENCE, folder)


def test_a_folder_with_the_evidence_and_records_of_every_candidate(tmp_path, sample_video):
    result = review(sample_video, tmp_path / "job", FakePlateModels())
    folder = tmp_path / "job"
    saved = json.loads((folder / "records.json").read_text())
    assert saved["schema_version"] == 2 and [r["event_id"] for r in saved["events"]] == ["evt-traffic_sample-wrong_way-f00048-t1", "evt-traffic_sample-red_light-f00070-t1"]
    assert sorted(p.name for p in (folder / "clips").iterdir()) == ["evt-traffic_sample-red_light-f00070-t1.mp4", "evt-traffic_sample-wrong_way-f00048-t1.mp4"]
    assert len(list((folder / "evidence").glob("*.jpg"))) == 6 and len(list((folder / "plates").glob("*.png"))) == 2
    for record in result.records:
        paths = [record["evidence"]["clip"], record["evidence"]["plate_crop"], *(f["image"] for f in record["evidence"]["frames"])]
        assert validate_record(record) == [] and all((folder / p).is_file() for p in paths) and record["review"]["status"] == "pending"
    assert json.loads((folder / "summary.json").read_text())["plates"] == {"unreadable": 2}  # The test car's plate is narrower than 30 px.


def test_red_light_images_show_the_stop_line(tmp_path, sample_video):
    result = review(sample_video, tmp_path, FakePlateModels())
    red = next(r for r in result.records if r["violation_type"] == "red_light")
    entry = cv2.imread(str(tmp_path / red["evidence"]["frames"][1]["image"])).astype(int)
    magenta = (entry[148:153, :, 0] > 200) & (entry[148:153, :, 1] < 80) & (entry[148:153, :, 2] > 200)
    assert magenta.any(axis=0).mean() > 0.8  # The line at y = 150 crosses the whole image, behind the label and the car.


def test_a_failed_plate_reading_keeps_every_record(tmp_path, sample_video):
    result = review(sample_video, tmp_path, BrokenPlateModels())
    assert len(result.records) == 2 and all(r["license_plate"]["status"] == "not_attempted" for r in result.records)
    assert all("plate_not_attempted" in r["needs_attention"] for r in result.records) and len(result.failures) == 2
    assert "The plate models' process stopped" in json.loads((tmp_path / "summary.json").read_text())["failures"][0]


def test_without_plate_models_the_records_say_so(tmp_path, sample_video):
    result = review(sample_video, tmp_path, None)
    assert result.failures == ["plate reading: no plate models"] and {r["license_plate"]["status"] for r in result.records} == {"not_attempted"}


def test_a_folder_already_in_use_is_refused(tmp_path, sample_video):
    (tmp_path / "records.json").write_text("{}")
    with pytest.raises(FileExistsError, match=r"already holds a review \(records.json\)"):
        review(sample_video, tmp_path, FakePlateModels())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["records.json"]


def test_a_folder_that_holds_only_the_uploaded_video_is_used(tmp_path, sample_video):
    (tmp_path / "video.mp4").write_bytes(sample_video.read_bytes())
    assert len(review(tmp_path / "video.mp4", tmp_path, FakePlateModels()).records) == 2 and (tmp_path / "records.json").is_file()


def test_a_file_where_the_folder_should_be_is_refused(tmp_path, sample_video):
    (tmp_path / "job").write_text("not a folder")
    with pytest.raises(FileExistsError, match="must be a folder"):
        review(sample_video, tmp_path / "job", FakePlateModels())


def test_no_candidate_gives_an_empty_list_of_records(tmp_path, sample_video):
    calm = {**CAMERA, "rules": {"wrong_way": {"scene": {**CAMERA["rules"]["wrong_way"]["scene"], "regions": [
        {**CAMERA["rules"]["wrong_way"]["scene"]["regions"][0], "allowed_direction": "toward_camera"}]}}}}
    result = review(sample_video, tmp_path, FakePlateModels(), calm)
    assert result.records == [] and json.loads((tmp_path / "records.json").read_text())["events"] == [] and not (tmp_path / "clips").exists()
