import pytest

from tvd.service.jobs import DONE, FAILED, INTERRUPTED, QUEUED, RUNNING, Job, JobStore


def job(job_id: str, created_at: str = "2026-11-07T10:00:00+00:00") -> Job:
    return Job(job_id, "cam1", "2026-11-07T17:00:00+07:00", "video.mp4", 60.0, 132.0, created_at)


def test_a_new_job_is_queued_and_read_back(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    assert store.get("a") == job("a") and store.get("a").state == QUEUED and store.get("missing") is None


def test_jobs_run_in_order_of_arrival(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("late", "2026-11-07T10:05:00+00:00"))
    store.add(job("early", "2026-11-07T10:01:00+00:00"))
    assert store.next_queued().job_id == "early" and [j.job_id for j in store.waiting()] == ["early", "late"]


def test_jobs_of_the_same_second_keep_the_order_they_were_added_in(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    for name in ("b", "c", "a"):
        store.add(job(name))
    assert [j.job_id for j in store.waiting()] == ["b", "c", "a"]


def test_a_job_runs_reports_its_progress_and_ends(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    store.start("a", "2026-11-07T10:01:00+00:00")
    store.progress("a", "tracking", 250, 1000)
    running = store.get("a")
    assert (running.state, running.stage, running.frames_done, running.frames_total) == (RUNNING, "tracking", 250, 1000) and store.next_queued() is None
    store.finish("a", "2026-11-07T10:03:00+00:00")
    assert (store.get("a").state, store.get("a").finished_at) == (DONE, "2026-11-07T10:03:00+00:00")


def test_a_failed_job_keeps_its_reason(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    store.start("a", "2026-11-07T10:01:00+00:00")
    store.fail("a", "RuntimeError: ffmpeg failed", "2026-11-07T10:02:00+00:00")
    assert (store.get("a").state, store.get("a").failure) == (FAILED, "RuntimeError: ffmpeg failed")


def test_a_state_only_moves_forward(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    with pytest.raises(ValueError, match="Job a is not running"):
        store.finish("a", "2026-11-07T10:01:00+00:00")
    store.start("a", "2026-11-07T10:01:00+00:00")
    with pytest.raises(ValueError, match="Job a is not queued"):
        store.start("a", "2026-11-07T10:02:00+00:00")


def test_after_a_restart_a_job_that_was_running_has_failed(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    store.add(job("b", "2026-11-07T10:01:00+00:00"))
    store.start("a", "2026-11-07T10:01:00+00:00")
    restarted = JobStore(tmp_path / "jobs.sqlite")
    assert restarted.interrupt("2026-11-07T11:00:00+00:00") == ["a"]
    assert (restarted.get("a").state, restarted.get("a").failure, restarted.get("b").state) == (FAILED, INTERRUPTED, QUEUED)


def test_old_finished_jobs_can_be_found_and_deleted(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    for name in ("old", "new", "waiting"):
        store.add(job(name))
    for name, end in (("old", "2026-10-01T10:00:00+00:00"), ("new", "2026-11-07T10:00:00+00:00")):
        store.start(name, end)
        store.finish(name, end)
    assert [j.job_id for j in store.finished_before("2026-11-01T00:00:00+00:00")] == ["old"]
    store.delete("old")
    assert store.get("old") is None and store.get("waiting").state == QUEUED
