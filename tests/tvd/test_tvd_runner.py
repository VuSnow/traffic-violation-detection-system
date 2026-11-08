import json

from conftest import FakePlateModels
from service_data import PLATES, copy_video, settings_for

from tvd.camera import load_rule_defaults
from tvd.plates import PLATE_MODELS, BasePlateModels
from tvd.service.cameras import CameraCatalog
from tvd.service.jobs import Job
from tvd.service.runner import Pipeline

JOB_ID = "20261107-103005-abcdef"


@PLATE_MODELS.register("test-one-use-plate")
class OneUse(FakePlateModels):
    """Models that are gone after they started, as after a call that did not answer in time."""

    def is_alive(self) -> bool:
        return False


@PLATE_MODELS.register("test-never-starts")
class NeverStarts(BasePlateModels):
    def __init__(self) -> None:
        raise RuntimeError("Plate models: no such model")

    def detect(self, image):
        return []

    def recognize(self, plate):
        raise NotImplementedError


def setup(tmp_path, sample_video, plates=PLATES):
    settings = settings_for(tmp_path, plates)
    camera = CameraCatalog(settings.files.cameras, load_rule_defaults(settings.files.rules), settings.jobs.seconds_per_video_second).get("test-camera")
    folder = tmp_path / "storage" / JOB_ID
    copy_video(sample_video, folder)
    job = Job(JOB_ID, "test-camera", "2026-11-07T08:00:00+07:00", "video.mp4", 10.0, 22.0, "2026-11-07T01:00:00+00:00")
    return Pipeline(settings.files), job, camera, folder


def test_a_job_is_reviewed_in_its_folder_with_the_job_id_in_every_record(tmp_path, sample_video):
    pipeline, job, camera, folder = setup(tmp_path, sample_video)
    reports = []
    result = pipeline.run(job, camera, folder, lambda *report: reports.append(report))
    records = json.loads((folder / "records.json").read_text())["events"]
    assert [r["event_id"] for r in records] == [f"evt-{JOB_ID}-wrong_way-f00048-t1", f"evt-{JOB_ID}-red_light-f00070-t1"] and len(result.records) == 2
    assert records[0]["track"]["tracking_session"] == f"{JOB_ID}/test-same-id" and records[0]["config_files"]["camera"] == camera.version
    assert set(records[0]["config_files"]) == {"perception", "rules", "plates", "evidence", "camera"} and (folder / "video.mp4").is_file()
    assert reports[0] == ("tracking", 0, 240) and reports[1] == ("tracking", 25, 240) and reports[-2:] == [("rules", 240, 240), ("evidence", 240, 240)]


def test_plate_models_that_are_gone_are_started_again(tmp_path, sample_video):
    pipeline, *_ = setup(tmp_path, sample_video, PLATES.replace("type: test-plate", "type: test-one-use-plate"))
    first = pipeline.plate_models()
    second = pipeline.plate_models()
    assert first is not second and first.closed


def test_without_plate_models_every_record_says_why(tmp_path, sample_video):
    pipeline, job, camera, folder = setup(tmp_path, sample_video, PLATES.replace("type: test-plate", "type: test-never-starts"))
    result = pipeline.run(job, camera, folder, lambda *report: None)
    assert result.failures == ["plate reading: the plate models did not start: Plate models: no such model"]
    assert {r["license_plate"]["status"] for r in result.records} == {"not_attempted"}
