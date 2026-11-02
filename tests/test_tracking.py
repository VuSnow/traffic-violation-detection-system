"""Tests for traffic_violation_detection.tracking. BoxMOT is replaced, except in the tests that name it."""

from pathlib import Path

import numpy as np
import pytest
from fakes import PassThroughTracker, ScriptedDetector

from traffic_violation_detection import tracking
from traffic_violation_detection.tracking import (
    add_to_tracks,
    collect_tracks,
    create_tracker,
    draw_tracks,
    join_track_fragments,
    majority_class,
    track_color,
    track_frames,
    track_video,
    update_trails,
    update_tracker,
    vehicle_detections,
)
from traffic_violation_detection.video import iter_frames, read_video_metadata

VIDEO = Path(__file__).resolve().parents[1] / "data" / "traffic_sample.mp4"  # 640x360, 24 FPS, 240 frames
CAR, TRUCK, PERSON, LIGHT = 2, 7, 0, 9


def detection(class_id, box=(10, 20, 110, 120), confidence=0.9):
    names = {CAR: "car", TRUCK: "truck", PERSON: "person", LIGHT: "traffic light"}
    return {"class_id": class_id, "class_name": names[class_id], "confidence": confidence, "bbox_xyxy": list(box)}


@pytest.fixture
def fake_tracker(monkeypatch):
    created = []

    def create():
        created.append(PassThroughTracker())
        return created[-1]

    monkeypatch.setattr(tracking, "create_tracker", create)
    return created


def test_only_vehicles_go_to_the_tracker():
    rows = vehicle_detections([detection(CAR), detection(PERSON), detection(LIGHT), detection(TRUCK)])
    assert rows.shape == (2, 6) and rows[:, 5].tolist() == [CAR, TRUCK]


def test_no_detections_gives_an_empty_tracker_input():
    assert vehicle_detections([]).shape == (0, 6)


def test_update_tracker_takes_the_class_from_the_matched_detection():
    tracked = update_tracker(PassThroughTracker(), [detection(CAR), detection(PERSON), detection(TRUCK)], None)
    assert [(t["track_id"], t["class_name"]) for t in tracked] == [(1, "car"), (2, "truck")]
    assert tracked[0]["bbox_xyxy"] == [10.0, 20.0, 110.0, 120.0]


def test_update_tracker_handles_a_frame_with_nothing_tracked():
    assert update_tracker(PassThroughTracker(), [], None) == []


def test_tracks_collect_rows_per_vehicle():
    tracks = {}
    tracked = update_tracker(PassThroughTracker(), [detection(CAR)], None)
    add_to_tracks(tracks, 5, tracked)
    add_to_tracks(tracks, 6, tracked)
    assert [row[0] for row in tracks["1"]] == [5, 6] and tracks["1"][0][5] == CAR


def test_collect_tracks_covers_a_range_of_frames(fake_tracker):
    detector = ScriptedDetector(lambda call: [(10, 20, 110, 120, 0.9, CAR)])
    tracks, detections = collect_tracks(VIDEO, detector, 10, 20)
    assert [row[0] for row in tracks["1"]] == list(range(10, 20)) and len(detections) == 10


def test_each_range_gets_a_new_tracker(fake_tracker):
    detector = ScriptedDetector()
    list(track_frames(VIDEO, detector, 0, 2))
    list(track_frames(VIDEO, detector, 2, 4))
    assert len(fake_tracker) == 2


def test_majority_class_ignores_a_single_wrong_frame():
    assert majority_class([[0, 0, 0, 1, 1, CAR, 0.9], [1, 0, 0, 1, 1, TRUCK, 0.9], [2, 0, 0, 1, 1, CAR, 0.9]]) == "car"


def test_track_color_is_stable_and_differs_between_tracks():
    assert track_color(7) == track_color(7) and track_color(7) != track_color(8)


def test_trails_follow_the_bottom_center_and_stay_short():
    trails = {}
    for k in range(5):
        update_trails(trails, [{"track_id": 3, "bbox_xyxy": [k, 0, k + 10, 20]}], length=3)
    assert trails[3] == [(7.0, 20), (8.0, 20), (9.0, 20)]


def test_draw_tracks_marks_the_box_and_the_trail():
    image = np.zeros((360, 640, 3), np.uint8)
    tracked = [{"track_id": 3, "bbox_xyxy": [100, 100, 200, 200]}]
    draw_tracks(image, tracked, {3: [(10.0, 300.0), (60.0, 300.0)]})
    assert image[200, 150].any() and image[300, 30].any() and not image[50, 400].any()


def test_track_video_saves_every_frame_under_the_expected_name(tmp_path, fake_tracker):
    detector = ScriptedDetector(lambda call: [(100, 100, 300, 250, 0.9, CAR)])
    path = track_video(VIDEO, tmp_path, model=detector)
    meta = read_video_metadata(path)
    assert path == tmp_path / "traffic_sample_output.mp4"
    assert (meta["width"], meta["height"]) == (640, 360) and meta["fps"] == pytest.approx(24.0)
    assert sum(1 for _ in iter_frames(path)) == detector.calls == 240


def test_track_video_fails_on_a_missing_file_before_loading_the_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "load_detector", lambda: pytest.fail("the detector was loaded"))
    with pytest.raises(FileNotFoundError):
        track_video(tmp_path / "missing.mp4", tmp_path)


def test_real_tracker_keeps_one_id_for_a_moving_vehicle():
    pytest.importorskip("boxmot")
    tracker = create_tracker()
    image = np.zeros((360, 640, 3), np.uint8)
    ids = set()
    for k in range(30):
        tracked = update_tracker(tracker, [detection(CAR, (100 + 4 * k, 100, 200 + 4 * k, 200))], image)
        ids |= {t["track_id"] for t in tracked}
    assert len(ids) == 1


def piece(first, last, x=100.0, step=0.0, height=100.0):
    # A track of a box whose bottom center moves step px per frame.
    return [[k, x + step * (k - first) - 20, 200, x + step * (k - first) + 20, 200 + height, 2, 0.9] for k in range(first, last + 1)]


def join(tracks, box_of=None):
    return join_track_fragments(tracks, max_gap_frames=6, max_jump_heights=0.1, max_size_ratio=1.25, box_of=box_of)


def test_join_merges_a_track_that_continues_another():
    joined = join({"0": piece(0, 10), "1": piece(12, 20)})
    assert list(joined) == ["0+1"] and [row[0] for row in joined["0+1"]] == list(range(0, 11)) + list(range(12, 21))


def test_join_keeps_one_row_per_frame_where_two_tracks_overlap():
    joined = join({"0": piece(0, 10), "1": piece(9, 20)})
    assert [row[0] for row in joined["0+1"]] == list(range(0, 21))


def test_join_leaves_tracks_too_far_apart_in_space():
    tracks = {"0": piece(0, 10), "1": piece(12, 20, x=120.0)}
    assert join(tracks) == tracks


def test_join_leaves_tracks_too_far_apart_in_time():
    tracks = {"0": piece(0, 10), "1": piece(17, 20)}
    assert join(tracks) == tracks


def test_join_leaves_boxes_of_different_sizes():
    tracks = {"0": piece(0, 10), "1": piece(12, 20, height=140.0)}
    assert join(tracks) == tracks


def test_join_leaves_two_vehicles_side_by_side():
    tracks = {"0": piece(0, 20), "1": piece(5, 25)}
    assert join(tracks) == tracks


def test_join_chains_three_parts():
    assert list(join({"0": piece(0, 5), "1": piece(6, 10), "2": piece(11, 15)})) == ["0+1+2"]


def test_join_picks_the_nearest_of_two_continuations():
    joined = join({"0": piece(0, 10), "1": piece(12, 20, x=108.0), "2": piece(12, 20, x=101.0)})
    assert sorted(joined) == ["0+2", "1"]


def test_join_measures_with_the_given_boxes():
    # On the image they are apart, but the camera moved, so on the road they are the same place.
    tracks = {"0": piece(0, 10), "1": piece(12, 20, x=160.0)}
    on_the_road = lambda row: [row[1] - (60 if row[0] >= 12 else 0), row[2], row[3] - (60 if row[0] >= 12 else 0), row[4]]
    assert list(join(tracks, on_the_road)) == ["0+1"]
