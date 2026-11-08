"""What the service writes in its log, from upload to removal. Events carry ids, states, counts and reasons, never a plate."""

import logging
from datetime import timedelta

import pytest
from service_data import PLATES, settings_for
from conftest import FakePlateModels
from test_tvd_runner import NeverStarts, OneUse  # noqa: F401  Registers "test-never-starts" and "test-one-use-plate".
from test_tvd_service import Clock, start, submit

from tvd.plates import PLATE_MODELS
from tvd.service.core import ReviewService
from tvd.service.intake import Refused
from tvd.service.logs import LOGGER
from tvd.service.runner import Pipeline
from tvd.service.worker import Worker


@pytest.fixture
def events(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return lambda: [(r.getMessage(), getattr(r, "fields", {})) for r in caplog.records if r.name == LOGGER]


def test_a_job_is_logged_from_its_upload_to_its_records(tmp_path, sample_video, events):
    service = start(tmp_path)
    job = submit(service, sample_video)
    service.run_next()
    logged = events()
    assert [name for name, _ in logged] == ["job_queued", "job_started", "job_stage", "job_stage", "job_stage", "job_done"]
    queued, started, *stages, done = (fields for _, fields in logged)
    assert queued == {"job_id": job.job_id, "camera_id": "test-camera", "video_seconds": 10.0, "estimate_seconds": 22.0, "queue_seconds": 0.0}
    assert started == {"job_id": job.job_id} and [s["stage"] for s in stages] == ["tracking", "rules", "evidence"]
    assert (done["job_id"], done["records"], done["plates"], done["failures"], done["estimate_seconds"]) == (job.job_id, 2, {"unreadable": 2}, [], 22.0)
    assert done["seconds"] >= 0


def test_no_log_line_holds_a_record_or_a_plate(tmp_path, sample_video, events):
    service = start(tmp_path)
    submit(service, sample_video)
    service.run_next()
    assert all("license_plate" not in fields and "text" not in fields and "events" not in fields for _, fields in events())


def test_a_refused_upload_is_logged_with_its_reason(tmp_path, sample_video, events):
    service = start(tmp_path)
    with pytest.raises(Refused):
        submit(service, sample_video, camera_id="cam9")
    assert events() == [("upload_refused", {"camera_id": "cam9", "status": 422, "reason": "Unknown camera 'cam9'. Known cameras: test-camera"})]


def test_a_failed_job_is_logged_with_its_reason(tmp_path, sample_video, events, caplog):
    service = start(tmp_path)
    job = submit(service, sample_video)
    (service.folders.folder(job.job_id) / "video.mp4").unlink()
    service.run_next()
    name, fields = events()[-1]
    failed = [r for r in caplog.records if r.getMessage() == "job_failed"][0]
    assert name == "job_failed" and fields["job_id"] == job.job_id and fields["reason"].startswith("FileNotFoundError: Video not found")
    assert failed.levelname == "ERROR" and failed.exc_info is not None


def test_a_restart_logs_the_jobs_it_interrupted_and_the_retention_logs_what_it_deletes(tmp_path, sample_video, events):
    clock = Clock()
    service = start(tmp_path, clock)
    running, finished = submit(service, sample_video), submit(service, sample_video)
    service.store.start(running.job_id, "2026-11-07T01:00:00+00:00")
    service.close()
    restarted = start(tmp_path, clock)
    restarted.run_next()
    clock.now += timedelta(days=8)
    restarted.remove_old()
    names = [(name, fields.get("job_id")) for name, fields in events()]
    assert ("job_interrupted", running.job_id) in names and ("job_removed", running.job_id) in names and ("job_removed", finished.job_id) in names


def test_plate_models_started_again_are_logged(tmp_path, sample_video, events):
    pipeline = Pipeline(settings_for(tmp_path, PLATES.replace("type: test-plate", "type: test-one-use-plate")).files)
    pipeline.plate_models()
    assert ("plate_models_restarted", {}) in events()


def test_a_worker_error_is_logged(events):
    from test_tvd_worker import Flaky, wait_for

    service = Flaky()
    worker = Worker(service, idle_seconds=0.05)
    worker.start()
    try:
        assert wait_for(lambda: service.calls >= 2, seconds=5)
    finally:
        worker.stop(timeout=5)
    assert ("worker_error", {"error": "RuntimeError: database is locked"}) in events()


@PLATE_MODELS.register("test-cpu-plate")
class OnTheCpu(FakePlateModels):
    """Models that had to start on the CPU, as when the GPU is full."""

    def describe(self):
        return {"type": "onnx_worker", "providers": ["CPUExecutionProvider"]}


def health_of(tmp_path, plates=PLATES, worker_running=True, **storage):
    settings = settings_for(tmp_path, plates)
    if storage:
        settings = settings.model_copy(update={"storage": settings.storage.model_copy(update=storage)})
    return ReviewService(settings, Clock()).health(worker_running)


def test_a_service_with_its_worker_and_room_is_healthy(tmp_path):
    report = health_of(tmp_path)
    assert (report["status"], report["problems"], report["worker_running"], report["queue"]) == ("ok", [], True, {"jobs": 0, "seconds": 0.0})
    assert report["started_at"] == "2026-11-07T01:00:00+00:00" and report["storage"]["free_gb"] > 0
    assert report["models"]["plates"] == {"type": "FakePlateModels"} and report["models"]["software"]["numpy"] == "2.4.6"


@pytest.mark.parametrize("change, problem", [
    ({"worker_running": False}, "the worker is not running: no job will be done"),
    ({"min_free_gb": 10**9}, "GB free in the storage, under the 1000000000.0 GB it needs"),
])
def test_a_service_that_cannot_do_jobs_is_failing(tmp_path, change, problem):
    report = health_of(tmp_path, **change)
    assert report["status"] == "failing" and any(problem in p for p in report["problems"])


@pytest.mark.parametrize("plates, problem", [
    (PLATES.replace("type: test-plate", "type: test-cpu-plate"), "the plate models run on the CPU: the GPU may be full"),
    (PLATES.replace("type: test-plate", "type: test-never-starts"), "no plates are read: the plate models did not start: Plate models: no such model"),
])
def test_a_service_that_does_jobs_less_well_is_degraded(tmp_path, plates, problem):
    report = health_of(tmp_path, plates)
    assert report["status"] == "degraded" and report["problems"] == [problem]
