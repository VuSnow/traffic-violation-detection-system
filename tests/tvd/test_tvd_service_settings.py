import pytest
from conftest import ROOT

from tvd.config import ConfigError
from tvd.service.settings import load_service_settings

SERVICE = ROOT / "configs" / "service.yaml"


def test_the_service_file_is_read():
    settings = load_service_settings(SERVICE)
    assert (settings.server.host, settings.server.port, settings.jobs.deadline_minutes) == ("127.0.0.1", 8040, 120)
    assert settings.files.cameras.as_posix() == "configs/cameras" and settings.storage.keep_days == 7 and settings.storage.min_free_gb == 10


def test_the_longest_video_follows_the_measured_speed():
    jobs = load_service_settings(SERVICE).jobs
    assert round(jobs.longest_video_seconds("fixed") / 60) == 55 and round(jobs.longest_video_seconds("orb") / 60) == 17
    assert jobs.work_seconds("orb", 60) == pytest.approx(420)


@pytest.mark.parametrize("old, new, message", [
    ("port: 8040", "port: 80400", "server.port: Input should be less than or equal to 65535"),
    ("    orb: 7.0", "    orb: 0", "jobs.seconds_per_video_second: Value error, every value must be greater than 0"),
    ("  keep_days: 7", "  keep_day: 7", "storage.keep_days: Field required"),
])
def test_a_wrong_setting_is_named(tmp_path, old, new, message):
    path = tmp_path / "service.yaml"
    path.write_text(SERVICE.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=message):
        load_service_settings(path)
