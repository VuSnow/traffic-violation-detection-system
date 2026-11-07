import numpy as np
import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_PLATES, notebook_08_plates, plate_read_from_json

from tvd.domain import Box
from tvd.plates import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateRead, load_plates_config, vote

READING = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml").reading
needs_notebook_08 = pytest.mark.skipif(not NOTEBOOK_08_PLATES.is_dir(), reason="Needs the outputs of Notebook 08, as on the GPU server.")


def plate_read(text: str, frame: int = 0, mean_prob: float = 0.99, rejected: str | None = None, width: int = 60) -> PlateRead:
    return PlateRead(frame, text, mean_prob, mean_prob, False, 0.9, width, 20, Box(0, 0, width, 20), rejected, np.zeros((20, width, 3), np.uint8))


def test_three_frames_that_agree_are_a_read():
    result = vote([plate_read("TEST1234", k) for k in range(3)], READING)
    assert (result.status, result.text, result.votes, result.usable_reads, result.agreement, result.mean_prob) == (READ, "TEST1234", 3, 3, 1.0, 0.99)


def test_two_of_three_frames_are_only_a_suggestion():
    result = vote([plate_read("TEST1234"), plate_read("TEST1235"), plate_read("TEST1234")], READING)
    assert (result.status, result.agreement, result.other_texts) == (UNCERTAIN, 0.667, {"TEST1235": 1})


def test_three_of_four_frames_are_a_read():
    assert vote([plate_read("TEST1234")] * 3 + [plate_read("TEST1235")], READING).status == READ


def test_one_usable_frame_is_only_a_suggestion():
    assert vote([plate_read("TEST1234"), plate_read("TES", rejected="too_short")], READING).status == UNCERTAIN


def test_a_plate_with_no_usable_read_is_unreadable():
    result = vote([plate_read("TES", rejected="too_short")], READING)
    assert (result.status, result.text, result.plates_found) == (UNREADABLE, None, 1)


def test_no_plate_found():
    result = vote([], READING)
    assert (result.status, result.text, result.best_read()) == (NO_PLATE, None, None)


def test_a_tie_goes_to_the_text_read_with_more_confidence():
    reads = [plate_read("TEST1234", mean_prob=0.96), plate_read("TEST1235", mean_prob=0.99)]
    assert vote(reads, READING).text == "TEST1235"


def test_the_evidence_is_the_largest_plate_of_the_chosen_text():
    reads = [plate_read("TEST1234", 1, width=50), plate_read("TEST1235", 2, width=90), plate_read("TEST1234", 3, width=70)]
    assert vote(reads, READING).best_read().frame == 3


def test_the_same_votes_as_the_notebooks():
    from traffic_violation_detection.plates import PLATE_SETTINGS, best_read, combine_reads

    rng = np.random.default_rng(1)
    for _ in range(300):
        reads = [
            plate_read(str(rng.choice(["TEST1234", "TEST1235", "FAKE9999"])), k, round(float(rng.uniform(0.9, 1.0)), 4),
                       rng.choice([None, None, None, "too_short", "low_confidence"]), int(rng.integers(30, 90)))
            for k in range(int(rng.integers(0, 8)))
        ]
        theirs = combine_reads([r.as_json() for r in reads], PLATE_SETTINGS)
        ours = vote(reads, READING)
        assert {k: v for k, v in ours.as_json().items() if k != "reads"} == theirs
        expected = best_read([r.as_json() for r in reads], theirs["text"])
        assert (ours.best_read().frame if ours.best_read() else None) == (expected["frame"] if expected else None)


@needs_notebook_08
def test_the_same_plates_as_notebook_08_from_its_reads():
    for event_id, saved in notebook_08_plates().items():
        expected = saved["license_plate"]
        result = vote([plate_read_from_json(r) for r in expected["reads"]], READING)
        matches = result.as_json() == expected  # Compared outside the assert, so that a failure never prints a real plate.
        assert matches, f"{event_id} differs from Notebook 08"
