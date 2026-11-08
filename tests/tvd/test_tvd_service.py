import io
from datetime import datetime, timedelta, timezone

import pytest
from service_data import settings_for

from tvd.service.core import NotFound, NotReady, ReviewService
from tvd.service.intake import Refused
from tvd.service.jobs import DONE, FAILED, INTERRUPTED, QUEUED

RECORDED_AT = "2026-11-07T08:00:00+07:00"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 7, 1, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def start(tmp_path, clock=None, **changes):
    settings = settings_for(tmp_path)
    if changes:
        settings = settings.model_copy(update={"jobs": settings.jobs.model_copy(update=changes)})
    return ReviewService(settings, clock or Clock())


def submit(service, video, camera_id="test-camera", recorded_at=RECORDED_AT, name="cam1 08h00.mp4"):
    with open(video, "rb") as upload:
        return service.submit(upload, name, camera_id, recorded_at)


def test_an_upload_becomes_a_queued_job_with_its_video_in_its_folder(tmp_path, sample_video):
    service = start(tmp_path)
    job = submit(service, sample_video)
    assert (job.state, job.camera_id, job.video_file, job.video_seconds, job.estimate_seconds) == (QUEUED, "test-camera", "video.mp4", 10.0, 22.0)
    assert job.job_id.startswith("20261107-010000-") and service.status(job.job_id) == job
    assert (service.folders.folder(job.job_id) / "video.mp4").read_bytes() == sample_video.read_bytes()


@pytest.mark.parametrize("change, reason, status", [
    ({"recorded_at": "2026-11-07T08:00:00"}, "needs a time zone", 422),
    ({"camera_id": "cam9"}, "Unknown camera 'cam9'. Known cameras: test-camera", 422),
])
def test_a_refused_upload_leaves_nothing_behind(tmp_path, sample_video, change, reason, status):
    service = start(tmp_path)
    with pytest.raises(Refused, match=reason) as refused:
        submit(service, sample_video, **change)
    assert refused.value.status == status and service.store.waiting() == [] and sorted(p.name for p in service.folders.root.iterdir()) == ["jobs.sqlite", "service.lock"]


def test_a_file_that_is_not_a_video_leaves_nothing_behind(tmp_path):
    service = start(tmp_path)
    with pytest.raises(Refused, match="could not open"):
        service.submit(io.BytesIO(b"not a video"), "notes.mp4", "test-camera", RECORDED_AT)
    assert service.store.waiting() == [] and sorted(p.name for p in service.folders.root.iterdir()) == ["jobs.sqlite", "service.lock"]


def test_an_upload_that_would_miss_the_deadline_is_refused(tmp_path, sample_video):
    service = start(tmp_path, deadline_minutes=0.6)  # 36 s: one job of 22 s fits, two do not.
    submit(service, sample_video)
    with pytest.raises(Refused, match="The queue is full: this video would be reviewed in about 1 min, after the deadline of 0.6 min") as refused:
        submit(service, sample_video)
    assert refused.value.status == 503 and len(service.store.waiting()) == 1


def test_a_job_runs_to_its_records_and_files(tmp_path, sample_video):
    service = start(tmp_path)
    job = submit(service, sample_video)
    with pytest.raises(NotReady, match="queued"):
        service.records(job.job_id)
    done = service.run_next()
    assert (done.state, done.stage, done.frames_done, done.frames_total, done.failure) == (DONE, "evidence", 240, 240, None)
    records = service.records(job.job_id)["events"]
    image = records[0]["evidence"]["frames"][0]["image"]
    assert len(records) == 2 and service.file(job.job_id, image).name.endswith("_before.jpg") and service.run_next() is None


@pytest.mark.parametrize("job_id, path", [("20261107-010000-abcdef", "records.json"), ("..", "records.json"), (None, "../jobs.sqlite"), (None, "nothing.jpg")])
def test_an_unknown_job_or_file_is_not_found(tmp_path, sample_video, job_id, path):
    service = start(tmp_path)
    job = submit(service, sample_video)
    with pytest.raises(NotFound):
        service.file(job_id or job.job_id, path)


def test_a_job_that_fails_keeps_its_reason_and_the_queue_goes_on(tmp_path, sample_video):
    service = start(tmp_path)
    broken, fine = submit(service, sample_video), submit(service, sample_video)
    (service.folders.folder(broken.job_id) / "video.mp4").unlink()
    first = service.run_next()
    assert (first.job_id, first.state) == (broken.job_id, FAILED) and first.failure.startswith("FileNotFoundError: Video not found")
    assert service.run_next().state == DONE and service.status(fine.job_id).state == DONE


def test_after_a_restart_the_job_that_was_running_has_failed_and_the_queue_is_kept(tmp_path, sample_video):
    service = start(tmp_path)
    running, waiting = submit(service, sample_video), submit(service, sample_video)
    service.store.start(running.job_id, "2026-11-07T01:00:00+00:00")
    service.close()
    restarted = start(tmp_path)
    assert restarted.interrupted == [running.job_id] and restarted.status(running.job_id).failure == INTERRUPTED
    assert restarted.status(waiting.job_id).state == QUEUED


def test_finished_jobs_older_than_the_retention_are_deleted_with_their_folders(tmp_path, sample_video):
    clock = Clock()
    service = start(tmp_path, clock)
    job = submit(service, sample_video)
    service.run_next()
    clock.now += timedelta(days=8)
    service.close()
    later = start(tmp_path, clock)
    with pytest.raises(NotFound):
        later.status(job.job_id)
    assert not later.folders.folder(job.job_id).exists()


def test_a_second_service_on_the_same_storage_is_refused(tmp_path):
    first = start(tmp_path)
    with pytest.raises(RuntimeError, match="Another service is using"):
        start(tmp_path)
    first.close()
    start(tmp_path).close()  # Once the first has stopped, the folder is free.
