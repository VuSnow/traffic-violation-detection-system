import io

import pytest
from conftest import ROOT

from tvd.camera import CameraConfig
from tvd.service.cameras import Camera
from tvd.service.intake import Refused, estimate_seconds, inspect_video, save_upload
from tvd.service.settings import JobSettings

SCENE = {"stop_line": [[0, 150], [640, 150]], "lane_dividers": [0, 600], "lane_direction_deg": 90, "traffic_direction": "toward_camera",
         "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}}
JOBS = JobSettings(deadline_minutes=120, seconds_per_video_second={"fixed": 2.2, "orb": 7.0})


def camera(frame_size=(640, 360), motion=None) -> Camera:
    config = {"camera_id": "test-camera", "frame_size": list(frame_size), "rules": {"red_light": {"scene": SCENE}}}
    if motion:
        config["motion"] = motion
    return Camera(CameraConfig.model_validate(config), ROOT / "test-camera.yaml", "sha256:000000000000")


def test_an_upload_is_saved_whole(tmp_path):
    assert save_upload(io.BytesIO(b"x" * 3_000_000), tmp_path / "job" / "video.mp4", 4_000_000) == 3_000_000
    assert (tmp_path / "job" / "video.mp4").stat().st_size == 3_000_000


def test_a_larger_upload_is_refused_and_nothing_is_kept(tmp_path):
    with pytest.raises(Refused, match="larger than 2 MB") as refused:
        save_upload(io.BytesIO(b"x" * 3_000_000), tmp_path / "video.mp4", 2 * 2**20)
    assert refused.value.status == 413 and not (tmp_path / "video.mp4").exists()


def test_an_empty_upload_is_refused(tmp_path):
    with pytest.raises(Refused, match="empty"):
        save_upload(io.BytesIO(b""), tmp_path / "video.mp4", 100)


def test_a_video_of_the_camera_is_described(sample_video):
    facts = inspect_video(sample_video, camera())
    assert (facts.frame_count, facts.fps, facts.width, facts.height, facts.seconds) == (240, 24.0, 640, 360, 10.0)


@pytest.mark.parametrize("change, reason", [
    ({"frame_size": (1920, 1080)}, "Camera test-camera has 1920x1080 frames, but traffic_sample.mp4 is 640x360"),
    ({"motion": {"type": "orb", "options": {"reference_frame": 500}}}, "reference frame 500 is beyond the video's 240 frames"),
])
def test_a_video_that_cannot_be_reviewed_with_this_camera_is_refused(sample_video, change, reason):
    with pytest.raises(Refused, match=reason) as refused:
        inspect_video(sample_video, camera(**change))
    assert refused.value.status == 422


def test_a_file_that_is_not_a_video_is_refused(tmp_path):
    (tmp_path / "video.mp4").write_text("not a video")
    with pytest.raises(Refused, match="could not open"):
        inspect_video(tmp_path / "video.mp4", camera())


def test_the_estimate_follows_the_measured_speed_of_the_camera(sample_video):
    facts = inspect_video(sample_video, camera())
    assert estimate_seconds(facts, camera(), JOBS) == pytest.approx(22.0)
    assert estimate_seconds(facts, camera(motion={"type": "orb", "options": {"reference_frame": 0}}), JOBS) == pytest.approx(70.0)


def test_a_video_too_long_for_the_deadline_is_refused(sample_video):
    short_deadline = JobSettings(deadline_minutes=0.3, seconds_per_video_second={"fixed": 2.2})
    with pytest.raises(Refused, match="lasts 0.2 min, and a video of a fixed camera can last 0.1 min at most, to be reviewed within 0.3 min"):
        estimate_seconds(inspect_video(sample_video, camera()), camera(), short_deadline)
