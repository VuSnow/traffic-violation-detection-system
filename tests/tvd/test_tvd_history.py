from tvd.domain import Box, TrackedObject
from tvd.tracking.history import TrackHistory


def tracked(track_id: int, x: float, class_name: str = "car") -> TrackedObject:
    return TrackedObject(track_id, Box(x, 0, x + 10, 10), 2, class_name, 0.9)


def test_objects_are_grouped_by_track_id():
    history = TrackHistory()
    history.add(0, [tracked(1, 0), tracked(2, 50)])
    history.add(1, [tracked(1, 5)])
    tracks = history.tracks
    assert sorted(tracks) == [1, 2] and history.frames_seen == 2
    assert (tracks[1].first_frame, tracks[1].last_frame, len(tracks[1].points)) == (0, 1, 2)


def test_a_frame_with_nothing_tracked_still_counts():
    history = TrackHistory()
    history.add(0, [])
    assert history.frames_seen == 1 and history.tracks == {}


def test_the_class_is_the_one_reported_most():
    history = TrackHistory()
    for index, name in enumerate(["car", "truck", "truck"]):
        history.add(index, [tracked(1, 0, name)])
    assert history.tracks[1].class_name == "truck"


def test_a_tie_keeps_the_class_reported_first():
    history = TrackHistory()
    history.add(0, [tracked(1, 0, "car")])
    history.add(1, [tracked(1, 0, "truck")])
    assert history.tracks[1].class_name == "car"


def test_box_at_a_frame():
    history = TrackHistory()
    history.add(3, [tracked(1, 7)])
    assert history.tracks[1].box_at(3) == Box(7, 0, 17, 10) and history.tracks[1].box_at(4) is None
