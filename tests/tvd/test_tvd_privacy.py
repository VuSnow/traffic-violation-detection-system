"""Real plate numbers must never reach a file that goes to GitHub. The private labels hold some; records and plates/ hold others."""

import json

import pytest
from conftest import ROOT

LABELS = ROOT / "data" / "private_annotations" / "plate_labels.json"
COMMITTED = ("src/tvd", "tests/tvd", "configs", "docs")


@pytest.mark.skipif(not LABELS.is_file(), reason="The private labels are only on machines that have them.")
def test_no_real_plate_appears_in_the_service_its_tests_settings_or_documents():
    videos = json.loads(LABELS.read_text())["videos"].values()
    real = {track["text"] for video in videos for track in video["tracks"].values() if "text" in track}
    files = [p for folder in COMMITTED for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".md", ".json"}]
    found = sorted(str(p.relative_to(ROOT)) for p in files if any(text in p.read_text(errors="ignore") for text in real))
    has_labels = len(real) > 0  # Asserted apart from the set: a failure shows file names, never a plate.
    assert has_labels and found == []
