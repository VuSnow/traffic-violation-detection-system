"""Tests for traffic_violation_detection.red_light on synthetic tracks and images."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_violation_detection.red_light import (
    AT_BORDER,
    CANDIDATE,
    COMPLIANT,
    ENDED,
    OUT_OF_RANGE,
    RIGHT_TURN,
    STOPPED,
    UNKNOWN_SIGNAL,
    build_event,
    deduplicate,
    evaluate_track,
    lane_of,
    line_y,
    load_red_light_scene,
    movement_after,
    read_group_state,
    read_head_state,
    read_lamp_state,
    smooth_states,
)

W, H, FPS = 1280, 720, 30.0
LANE_ANGLE = -24.0
SCENE_PATH = Path(__file__).resolve().parents[1] / "configs" / "scenes" / "rlr_occasions.json"
CLIP = {
    "start": 0,
    "end": 400,
    "stop_line": [[100, 400], [1280, 400]],
    "lane_dividers": [300, 560, 830, 1070],
    "evaluable_x_max": 1280,
}


def states(value: str, n: int = 400) -> dict:
    return {"through": {k: value for k in range(n)}, "left": {k: value for k in range(n)}}


def track(start, velocity, n=120, height=100, width=120, first_frame=0, stop_after=None) -> dict:
    # A vehicle moving at constant velocity; (x, y) is its bottom center. It may stop after some frames.
    frames = np.arange(first_frame, first_frame + n)
    steps = np.arange(n) if stop_after is None else np.minimum(np.arange(n), stop_after)
    x = start[0] + steps * velocity[0]
    y = start[1] + steps * velocity[1]
    return {"track_id": "1", "class_name": "car", "frames": frames, "boxes": np.c_[x - width / 2, y - height, x + width / 2, y]}


def judge(trk, signal=None, clip=CLIP):
    return evaluate_track(trk, clip, signal or states("red"), LANE_ANGLE, W, H)


# Up and to the right, along the lane direction of about -24 degrees.
ALONG_LANE = (2.74, -1.22)


def test_vehicle_passing_on_red_is_a_candidate():
    assert judge(track((600, 480), ALONG_LANE))["verdict"] == CANDIDATE


def test_vehicle_passing_on_green_is_not_flagged():
    assert judge(track((600, 480), ALONG_LANE), states("green"))["verdict"] == COMPLIANT


def test_vehicle_entering_on_yellow_is_not_flagged():
    assert judge(track((600, 480), ALONG_LANE), states("yellow"))["verdict"] == COMPLIANT


@pytest.mark.parametrize("value", ["unknown", "conflict"])
def test_unreadable_signal_is_not_evaluable_not_a_violation(value):
    assert judge(track((600, 480), ALONG_LANE), states(value))["verdict"] == UNKNOWN_SIGNAL


def test_missing_signal_frames_are_not_evaluable():
    assert judge(track((600, 480), ALONG_LANE), {"through": {}, "left": {}})["verdict"] == UNKNOWN_SIGNAL


def test_vehicle_stopping_on_the_line_is_not_flagged():
    # The front passes the line (entry offset), but the rear never does.
    record = judge(track((600, 480), ALONG_LANE, stop_after=50))
    assert record["verdict"] == STOPPED and not record["passed_line"]


def test_vehicle_stopping_before_the_line_never_crosses():
    assert judge(track((600, 480), ALONG_LANE, stop_after=20)) is None


def test_right_turn_on_red_is_not_flagged():
    # Follows the lane across the line, then turns to the right of the lane direction.
    frames = np.arange(140)
    x = 900 + 3.0 * frames
    y = 450 - 1.0 * np.minimum(frames, 50) + 1.5 * np.maximum(frames - 50, 0)
    moving = {"track_id": "1", "class_name": "car", "frames": frames, "boxes": np.c_[x - 60, y - 100, x + 60, y]}
    record = judge(moving)
    assert record["movement"] == "right" and record["verdict"] == RIGHT_TURN


def test_box_at_the_image_border_is_not_evaluable():
    assert judge(track((600, 480), ALONG_LANE, width=1500))["verdict"] == AT_BORDER


def test_crossing_behind_an_overlay_is_not_evaluable():
    clip = {**CLIP, "evaluable_x_max": 500}
    assert judge(track((600, 480), ALONG_LANE), clip=clip)["verdict"] == OUT_OF_RANGE


def test_track_ending_right_after_the_entry_is_not_evaluable():
    assert judge(track((600, 434), ALONG_LANE, n=10))["verdict"] == ENDED


def test_vehicle_still_moving_when_its_track_ends_is_not_called_stopped():
    # Enough frames after the entry, but the rear has not passed by the margin and the vehicle is still moving.
    assert judge(track((600, 480), ALONG_LANE, n=70))["verdict"] == ENDED


def test_clip_ending_right_after_the_entry_is_not_evaluable():
    stationary = track((600, 480), ALONG_LANE, n=140, stop_after=50)
    entry = judge(stationary)["frame"]
    assert judge(stationary, clip={**CLIP, "end": entry + 5})["verdict"] == ENDED


def test_fragment_whose_rear_never_crosses_gives_no_record():
    # The front is past the line from the first frame and the rear stays before it: no crossing was observed.
    assert judge(track((600, 425), ALONG_LANE, n=10)) is None


def test_track_too_short_gives_no_record():
    assert judge(track((600, 480), ALONG_LANE, n=5)) is None


def test_fragment_starting_past_the_entry_is_still_judged():
    # The front is already past the line when the track starts, and the rear crosses it later.
    record = judge(track((600, 420), ALONG_LANE))
    assert record["verdict"] == CANDIDATE and not record["entry_seen"]


def test_left_lane_uses_the_left_signal():
    signal = {"through": {k: "green" for k in range(400)}, "left": {k: "red" for k in range(400)}}
    record = judge(track((150, 480), ALONG_LANE), signal)
    assert record["lane"] == "left_turn" and record["signal_group"] == "left" and record["verdict"] == CANDIDATE
    assert judge(track((600, 480), ALONG_LANE), signal)["verdict"] == COMPLIANT


def test_state_duration_counts_back_from_the_entry():
    signal = {"through": {k: "green" if k < 10 else "red" for k in range(400)}, "left": {}}
    record = judge(track((600, 480), ALONG_LANE), signal)
    assert record["state_for_frames"] == record["frame"] - 10


def test_line_y_interpolates():
    assert line_y([[0, 100], [100, 200]], 50) == pytest.approx(150)


def test_lanes_follow_the_dividers():
    assert [lane_of(CLIP, x) for x in (200, 600, 1100)] == ["left_turn", "through", "right_turn"]


@pytest.mark.parametrize(
    "displacement, expected",
    [((100, -45), "straight"), ((100, 40), "right"), ((-20, -100), "left"), ((10, -10), "unknown")],
)
def test_movement_is_relative_to_the_lane(displacement, expected):
    assert movement_after(displacement, LANE_ANGLE) == expected


def test_smooth_states_removes_flicker():
    assert smooth_states(["red"] * 5 + ["unknown"] + ["red"] * 5) == ["red"] * 11


def glowing_patch(bgr, size=40) -> np.ndarray:
    # A saturated white center with a coloured glow, as at night.
    patch = np.zeros((size, size, 3), np.uint8)
    cv2.circle(patch, (size // 2, size // 2), size // 2 - 2, bgr, -1)
    cv2.circle(patch, (size // 2, size // 2), 5, (255, 255, 255), -1)
    return patch


@pytest.mark.parametrize("bgr, expected", [((40, 40, 230), "red"), ((30, 200, 240), "yellow"), ((230, 220, 40), "green")])
def test_head_state_reads_the_glow(bgr, expected):
    assert read_head_state(glowing_patch(bgr)) == expected


def test_monochrome_head_is_unknown():
    assert read_head_state(np.full((40, 40, 3), 200, np.uint8)) == "unknown"


def test_dark_head_is_unknown():
    assert read_head_state(np.zeros((40, 40, 3), np.uint8)) == "unknown"


LAMPS = {"red": [50, 30], "yellow": [50, 50], "green": [50, 70]}


@pytest.mark.parametrize("colour, bgr", [("red", (30, 30, 230)), ("yellow", (20, 210, 240)), ("green", (80, 220, 60))])
def test_lamp_state_reads_each_position(colour, bgr):
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(frame, tuple(LAMPS[colour]), 7, bgr, -1)
    assert read_lamp_state(frame, LAMPS) == colour


def test_unlit_lamps_are_unknown():
    assert read_lamp_state(np.zeros((100, 100, 3), np.uint8), LAMPS) == "unknown"


def test_lamp_off_the_image_is_ignored():
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(frame, (50, 50), 7, (20, 210, 240), -1)
    assert read_lamp_state(frame, {"red": [50, 2], "yellow": [50, 50], "green": [50, 70]}) == "yellow"


def test_group_state_reports_a_conflict():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[10:50, 10:50] = glowing_patch((40, 40, 230))
    frame[10:50, 110:150] = glowing_patch((230, 220, 40))
    heads = [[10, 10, 50, 50], [110, 10, 150, 50]]
    assert read_group_state(frame, "head", heads) == "conflict"
    assert read_group_state(frame, "head", heads[:1]) == "red"


def candidate(frame, x):
    return {"verdict": CANDIDATE, "frame": frame, "x": x}


def test_deduplicate_merges_nearby_candidates_only():
    records = [candidate(100, 500), candidate(110, 540), candidate(200, 500), candidate(110, 900)]
    assert [(r["frame"], r["x"]) for r in deduplicate(records, FPS)] == [(100, 500), (110, 900), (200, 500)]


def test_event_is_pending_and_its_id_differs_from_the_track_id():
    record = judge(track((600, 480), ALONG_LANE))
    event = build_event(record, "B", "rlr_occasions", FPS, clip_start=0)
    assert event["review_status"] == "pending" and event["license_plate"] is None
    assert event["event_id"] != event["track_id"] and "rlr_occasions" in event["event_id"]
    json.dumps(event)


def test_committed_scene_loads_and_covers_all_clips():
    scene = load_red_light_scene(SCENE_PATH, W, H)
    assert len(scene["clips"]) == 18
    assert {name for name, clip in scene["clips"].items() if clip["role"] == "development"} == {"B", "F", "H"}


def test_scene_of_another_size_is_rejected():
    with pytest.raises(ValueError, match="video is 1920x1080"):
        load_red_light_scene(SCENE_PATH, 1920, 1080)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.update(stop_line=[[0, 0]]), "stop line"),
        (lambda c: c.update(stop_line=[[100, 0], [50, 0]]), "stop line"),
        (lambda c: c.update(start=10, end=5), "empty frame range"),
        (lambda c: c["signals"].update(method="magic"), "unknown signal method"),
        (lambda c: c["signals"].update(through=[]), "no signal"),
    ],
)
def test_broken_scenes_are_rejected(tmp_path, mutate, message):
    scene = json.loads(SCENE_PATH.read_text())
    broken = copy.deepcopy(scene)
    mutate(broken["clips"]["A"])
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match=message):
        load_red_light_scene(path, W, H)
