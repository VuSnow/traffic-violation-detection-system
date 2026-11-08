import os
from datetime import datetime, timezone

import pytest

from tvd.service.storage import JobFolders

NOW = datetime(2026, 11, 7, 10, 30, 5, tzinfo=timezone.utc)


def test_a_job_id_starts_with_the_upload_time(tmp_path):
    folders = JobFolders(tmp_path / "storage")
    first, second = folders.new_id(NOW), folders.new_id(NOW)
    assert first.startswith("20261107-103005-") and len(first) == 22 and first != second
    assert folders.folder(first) == tmp_path / "storage" / first and folders.database == tmp_path / "storage" / "jobs.sqlite"


@pytest.mark.parametrize("job_id", ["..", "../storage", "20261107-103005-abc", "20261107-103005-abcdef/.."])
def test_anything_but_a_job_id_is_refused(tmp_path, job_id):
    with pytest.raises(ValueError, match="Not a job id"):
        JobFolders(tmp_path).folder(job_id)


def test_only_files_inside_the_job_folder_are_given(tmp_path):
    folders = JobFolders(tmp_path / "storage")
    job_id = folders.new_id(NOW)
    folder = folders.folder(job_id)
    (folder / "evidence").mkdir(parents=True)
    (folder / "evidence" / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "secret.txt").write_text("not for the backend")
    os.symlink(tmp_path / "secret.txt", folder / "evidence" / "link.jpg")
    assert folders.file(job_id, "evidence/a.jpg") == (folder / "evidence" / "a.jpg").resolve()
    assert [folders.file(job_id, p) for p in ("../../secret.txt", "evidence/link.jpg", "evidence", "missing.jpg")] == [None] * 4


def test_a_job_folder_is_removed(tmp_path):
    folders = JobFolders(tmp_path)
    job_id = folders.new_id(NOW)
    folders.folder(job_id).mkdir()
    folders.remove(job_id)
    assert not folders.folder(job_id).exists()
