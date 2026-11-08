import json
import logging

import pytest
from service_data import settings_for

from tvd.config import ConfigError
from tvd.service.logs import LOGGER, log_event, setup_logging
from tvd.service.settings import LoggingSettings


@pytest.fixture(autouse=True)
def restore_logger():
    yield
    logger = logging.getLogger(LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_service_file_says_where_and_how_much_to_log(tmp_path):
    settings = settings_for(tmp_path).logging
    assert (settings.folder.name, settings.level, settings.max_mb, settings.keep_files) == ("logs", "INFO", 20, 5)


def test_a_wrong_level_is_named(tmp_path):
    from service_data import service_files
    from tvd.service.settings import load_service_settings

    path = service_files(tmp_path)
    path.write_text(path.read_text().replace("level: INFO", "level: LOUD"))
    with pytest.raises(ConfigError, match="logging.level: Input should be 'DEBUG', 'INFO', 'WARNING' or 'ERROR'"):
        load_service_settings(path)


def test_each_event_is_one_json_line_in_the_file_and_on_standard_output(tmp_path, capsys):
    path = setup_logging(LoggingSettings(folder=tmp_path / "logs", level="INFO", max_mb=1, keep_files=2))
    log_event("job_submitted", job_id="20261108-100314-fb185c", camera_id="cam1", estimate_seconds=190.1)
    (line,) = lines(path)
    assert path == tmp_path / "logs" / "service.log" and line["event"] == "job_submitted" and line["level"] == "INFO"
    assert (line["job_id"], line["camera_id"], line["estimate_seconds"]) == ("20261108-100314-fb185c", "cam1", 190.1) and line["time"].endswith("+00:00")
    assert json.loads(capsys.readouterr().out.strip()) == line


def test_an_error_keeps_its_traceback(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="INFO", max_mb=1, keep_files=2))
    try:
        raise RuntimeError("ffmpeg failed")
    except RuntimeError:
        log_event("job_failed", logging.ERROR, job_id="x", exc_info=True)
    (line,) = lines(path)
    assert line["level"] == "ERROR" and "RuntimeError: ffmpeg failed" in line["traceback"] and "Traceback" in line["traceback"]


def test_an_error_field_and_a_traceback_are_both_kept(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="INFO", max_mb=1, keep_files=2))
    try:
        raise RuntimeError("database is locked")
    except RuntimeError:
        log_event("worker_error", logging.ERROR, exc_info=True, error="RuntimeError: database is locked")
    (line,) = lines(path)
    assert line["error"] == "RuntimeError: database is locked" and "Traceback" in line["traceback"]


def test_events_below_the_level_are_left_out(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="WARNING", max_mb=1, keep_files=2))
    log_event("job_progress", stage="tracking")
    log_event("worker_error", logging.WARNING, error="database is locked")
    assert [line["event"] for line in lines(path)] == ["worker_error"]


def test_a_full_file_is_rotated_and_only_so_many_are_kept(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="INFO", max_mb=0.001, keep_files=2))  # About 1 kB per file.
    for k in range(100):
        log_event("job_progress", job_id="x", frames_done=k)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["service.log", "service.log.1", "service.log.2"]


def test_starting_twice_does_not_write_each_line_twice(tmp_path):
    settings = LoggingSettings(folder=tmp_path, level="INFO", max_mb=1, keep_files=2)
    setup_logging(settings)
    path = setup_logging(settings)
    log_event("service_started")
    assert len(lines(path)) == 1
