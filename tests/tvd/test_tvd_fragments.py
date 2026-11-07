import pytest
from notebook_data import OUTPUTS, history_from_rows, load_json

from tvd.domain import Box
from tvd.tracking.fragments import join_fragments
from tvd.tracking.history import TrackPoint

STORED_TRACKS = OUTPUTS / "05b_red_light_violation" / "tracks.json"
LIMITS = {"max_gap_frames": 6, "max_jump_heights": 0.1, "max_size_ratio": 1.25}


def points(first: int, last: int, x: float = 100, height: float = 40) -> list[TrackPoint]:
    """A vehicle standing still, seen from frame ``first`` to ``last``."""
    return [TrackPoint(k, Box(x, 100, x + 50, 100 + height), 0.9, "car") for k in range(first, last + 1)]


def test_a_track_that_continues_another_is_joined():
    joined = join_fragments({"3": points(0, 10), "8": points(12, 20)}, **LIMITS)
    assert list(joined) == ["3+8"] and [p.frame_index for p in joined["3+8"]] == list(range(11)) + list(range(12, 21))


@pytest.mark.parametrize("second, why", [
    (points(20, 30), "starts too long after"),
    (points(12, 20, x=110), "starts too far away"),
    (points(12, 20, height=60), "is a different size"),
])
def test_tracks_that_may_be_two_vehicles_are_kept_apart(second, why):
    assert list(join_fragments({"3": points(0, 10), "8": second}, **LIMITS)) == ["3", "8"], why


def test_the_closest_continuation_is_joined_first():
    joined = join_fragments({"1": points(0, 10), "2": points(11, 20, x=103), "3": points(11, 20, x=101)}, **LIMITS)
    assert sorted(joined) == ["1+3", "2"]


@pytest.mark.skipif(not STORED_TRACKS.is_file(), reason="Needs the outputs of Notebook 05b, as on the GPU server.")
def test_the_same_joins_as_the_notebooks():
    from traffic_violation_detection.tracking import join_track_fragments

    for clip, tracks in load_json(STORED_TRACKS).items():
        expected = join_track_fragments(tracks, **LIMITS)
        ours = join_fragments({str(t.track_id): t.points for t in history_from_rows(tracks).tracks.values()}, **LIMITS)
        assert {k: [p.frame_index for p in v] for k, v in ours.items()} == {k: [int(r[0]) for r in v] for k, v in expected.items()}, clip
        assert list(ours) == list(expected), clip
