import time

from service_data import settings_for

from tvd.service.core import ReviewService
from tvd.service.jobs import DONE
from tvd.service.worker import Worker


def wait_for(condition, seconds: float = 60.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


def test_the_worker_runs_queued_jobs_in_order_until_stopped(tmp_path, sample_video):
    service = ReviewService(settings_for(tmp_path))
    with open(sample_video, "rb") as upload:
        first = service.submit(upload, "a.mp4", "test-camera", "2026-11-07T08:00:00+07:00")
    worker = Worker(service, idle_seconds=0.1)
    worker.start()
    try:
        assert wait_for(lambda: service.status(first.job_id).state == DONE)
        with open(sample_video, "rb") as upload:
            second = service.submit(upload, "b.mp4", "test-camera", "2026-11-07T08:05:00+07:00")
        assert wait_for(lambda: service.status(second.job_id).state == DONE)  # A job that arrives later is picked up too.
        assert service.status(first.job_id).finished_at <= service.status(second.job_id).started_at
    finally:
        worker.stop(timeout=30)
    assert not worker.running


def test_a_stopped_worker_takes_no_more_jobs(tmp_path, sample_video):
    service = ReviewService(settings_for(tmp_path))
    worker = Worker(service, idle_seconds=0.1)
    worker.start()
    worker.stop(timeout=30)
    with open(sample_video, "rb") as upload:
        job = service.submit(upload, "a.mp4", "test-camera", "2026-11-07T08:00:00+07:00")
    time.sleep(0.5)
    assert not worker.running and service.status(job.job_id).state == "queued"


class Flaky:
    """A service whose first call fails, as with a database locked for a moment."""

    def __init__(self) -> None:
        self.calls = 0

    def run_next(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("database is locked")
        return None

    def remove_old(self):
        return []


def test_the_worker_outlives_an_unexpected_error():
    service = Flaky()
    worker = Worker(service, idle_seconds=0.05)
    worker.start()
    try:
        assert wait_for(lambda: service.calls >= 3, seconds=5)
    finally:
        worker.stop(timeout=5)
    assert worker.errors == ["RuntimeError: database is locked"]
