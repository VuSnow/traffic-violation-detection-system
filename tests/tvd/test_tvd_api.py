import time

import pytest
from fastapi.testclient import TestClient
from service_data import settings_for

from tvd.service.api import create_app
from tvd.service.core import ReviewService

FORM = {"camera_id": "test-camera", "recorded_at": "2026-11-07T08:00:00+07:00"}


@pytest.fixture
def service(tmp_path):
    return ReviewService(settings_for(tmp_path))


@pytest.fixture
def client(service):
    return TestClient(create_app(service, run_worker=False))


def upload(client, video, **form):
    with open(video, "rb") as data:
        return client.post("/jobs", files={"video": ("cam1.mp4", data, "video/mp4")}, data={**FORM, **form})


def test_the_cameras_are_listed(client):
    assert client.get("/cameras").json() == [{"camera_id": "test-camera", "frame_size": [640, 360], "motion": "fixed", "moving": False, "rules": ["red_light", "wrong_way"]}]


def test_an_upload_is_accepted_with_where_to_follow_it(client, sample_video):
    response = upload(client, sample_video)
    job = response.json()
    assert response.status_code == 202 and job["state"] == "queued" and job["status_url"] == f"/jobs/{job['job_id']}"
    assert client.get(job["status_url"]).json()["estimate_seconds"] == 22.0


@pytest.mark.parametrize("form, status, detail", [
    ({"recorded_at": "2026-11-07T08:00:00"}, 422, "needs a time zone"),
    ({"camera_id": "cam9"}, 422, "Unknown camera 'cam9'"),
])
def test_a_refused_upload_says_why(client, sample_video, form, status, detail):
    response = upload(client, sample_video, **form)
    assert response.status_code == status and detail in response.json()["detail"]


def test_an_upload_without_a_camera_is_refused_by_the_api_itself(client, sample_video):
    with open(sample_video, "rb") as data:
        response = client.post("/jobs", files={"video": ("cam1.mp4", data, "video/mp4")}, data={"recorded_at": FORM["recorded_at"]})
    assert response.status_code == 422 and response.json()["detail"][0]["loc"] == ["body", "camera_id"]


def test_an_upload_larger_than_the_limit_is_refused_before_it_is_read(tmp_path):
    settings = settings_for(tmp_path)
    small = settings.model_copy(update={"storage": settings.storage.model_copy(update={"max_upload_mb": 1})})
    client = TestClient(create_app(ReviewService(small), run_worker=False))
    response = client.post("/jobs", files={"video": ("big.mp4", b"x" * (2 * 2**20), "video/mp4")}, data=FORM)
    assert response.status_code == 413 and response.json()["detail"] == "The upload is larger than 1 MB"


def test_records_come_when_the_job_is_done_and_its_files_can_be_fetched(client, service, sample_video):
    job_id = upload(client, sample_video).json()["job_id"]
    early = client.get(f"/jobs/{job_id}/records")
    assert early.status_code == 409 and early.json()["detail"] == f"Job {job_id} is queued: its records come when it is done"
    service.run_next()
    records = client.get(f"/jobs/{job_id}/records").json()["events"]
    image = client.get(f"/jobs/{job_id}/files/{records[1]['evidence']['frames'][1]['image']}")
    assert len(records) == 2 and image.status_code == 200 and image.headers["content-type"] == "image/jpeg" and image.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize("path", ["/jobs/20261107-010000-abcdef", "/jobs/../cameras/records", "/jobs/{job}/files/..%2Fjobs.sqlite", "/jobs/{job}/files/nothing.jpg"])
def test_an_unknown_job_or_file_is_not_found(client, sample_video, path):
    job_id = upload(client, sample_video).json()["job_id"]
    assert client.get(path.replace("{job}", job_id)).status_code == 404


def test_with_its_worker_the_app_runs_each_upload_to_its_records(service, sample_video):
    with TestClient(create_app(service)) as client:
        job_id = upload(client, sample_video).json()["job_id"]
        end = time.monotonic() + 60
        while client.get(f"/jobs/{job_id}").json()["state"] in ("queued", "running") and time.monotonic() < end:
            time.sleep(0.2)
        assert client.get(f"/jobs/{job_id}").json()["state"] == "done" and len(client.get(f"/jobs/{job_id}/records").json()["events"]) == 2


def test_health_is_failing_without_a_worker(client):
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["problems"] == ["the worker is not running: no job will be done"]


def test_health_is_ok_while_the_app_runs_its_worker(service):
    with TestClient(create_app(service)) as client:
        response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"
