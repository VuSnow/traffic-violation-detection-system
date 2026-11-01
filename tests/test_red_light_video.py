"""Tests for the function from a video to red-light results, on a small drawn video with stand-in detector and tracker."""

import json

import cv2
import numpy as np
import pytest
from fakes import PassThroughTracker, ScriptedDetector, write_video

from traffic_violation_detection import red_light, tracking
from traffic_violation_detection.red_light import CANDIDATE, COMPLIANT, UNKNOWN_SIGNAL, as_track, detect_red_light_video, read_frame_signals, track_and_read_clip
from traffic_violation_detection.video import iter_frames

WIDTH, HEIGHT = 320, 240
COLORS = {"red": (30, 30, 230), "green": (230, 220, 40)}
LINE_Y = 120


def head_frame(color):
    image = np.full((HEIGHT, WIDTH, 3), 40, np.uint8)
    if color:
        cv2.rectangle(image, (20, 20), (60, 60), COLORS[color], -1)
        cv2.rectangle(image, (260, 20), (300, 60), COLORS[color], -1)
    return image


def clip_config(start=0, end=60):
    return {
        "start": start, "end": end, "lighting": "night", "role": "evaluation",
        "stop_line": [[0, LINE_Y], [WIDTH, LINE_Y]], "lane_dividers": [50, 280], "evaluable_x_max": WIDTH,
        "signals": {"method": "head", "through": [[20, 20, 60, 60]], "left": [[260, 20, 300, 60]]},
    }


def write_scene(tmp_path, clips):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene_id": "small", "video": {"file": "small.mp4", "width": WIDTH, "height": HEIGHT, "fps": 30.0},
        "lane_direction_deg": -90.0, "clips": clips,
    }))
    return path


def car_going_up(call):
    # Bottom edge from y=220 upward at 3 px per frame: its front crosses the line near call 29, and its rear goes well past.
    bottom = 220 - 3 * call
    return [(140, bottom - 40, 180, bottom, 0.9, 2)] if bottom > 45 else []


@pytest.fixture
def trackers(monkeypatch):
    created = []

    def create():
        created.append(PassThroughTracker())
        return created[-1]

    monkeypatch.setattr(tracking, "create_tracker", create)
    return created


def run(tmp_path, color, clips=None, script=car_going_up):
    video = write_video(tmp_path / "small.mp4", [head_frame(color)] * 60)
    return detect_red_light_video(video, tmp_path / "out", write_scene(tmp_path, clips or {"main": clip_config()}), model=ScriptedDetector(script))


def test_car_on_red_gives_one_candidate_and_every_output(tmp_path, trackers):
    result = run(tmp_path, "red")
    assert [e["clip"] for e in result["events"]] == ["main"] and result["crossings"]["main"][0]["verdict"] == CANDIDATE
    assert result["output_video"] == tmp_path / "out" / "small_output.mp4"
    assert sum(1 for _ in iter_frames(result["output_video"])) == 60
    manifest = json.loads(result["clips_manifest"].read_text())
    entry = manifest["clips"][0]
    assert (result["clips_manifest"].parent / entry["clip_file"]).is_file()
    assert entry["signal_source"] == "camera" and entry["review_status"] == "pending" and entry["license_plate"] is None
    saved = json.loads(result["events_file"].read_text())
    assert saved["events"][0]["review_status"] == "pending" and saved["signal_source"] == "camera"


def test_car_on_green_is_not_flagged(tmp_path, trackers):
    result = run(tmp_path, "green")
    assert result["events"] == [] and result["crossings"]["main"][0]["verdict"] == COMPLIANT
    assert json.loads(result["clips_manifest"].read_text())["clips"] == []


def test_unreadable_signal_is_not_evaluable_and_not_a_violation(tmp_path, trackers):
    result = run(tmp_path, None)
    assert result["events"] == [] and result["crossings"]["main"][0]["verdict"] == UNKNOWN_SIGNAL


def test_only_the_frames_of_the_clips_are_analysed_and_saved(tmp_path, trackers):
    detector = ScriptedDetector(car_going_up)
    video = write_video(tmp_path / "small.mp4", [head_frame("red")] * 60)
    result = detect_red_light_video(video, tmp_path / "out", write_scene(tmp_path, {"main": clip_config(10, 50)}), model=detector)
    assert detector.calls == 40 and sum(1 for _ in iter_frames(result["output_video"])) == 40


def test_each_clip_gets_its_own_tracker(tmp_path, trackers):
    result = run(tmp_path, "green", {"a": clip_config(0, 20), "b": clip_config(20, 40)})
    assert len(trackers) == 2 and sum(1 for _ in iter_frames(result["output_video"])) == 40


def test_candidate_is_marked_from_its_entry_in_the_result_video(tmp_path, trackers):
    result = run(tmp_path, "red")
    frames = [r["image"] for r in iter_frames(result["output_video"])]
    entry = result["crossings"]["main"][0]["frame"]
    # A red box around the car, away from the red signal heads.
    red = lambda image: int(((image[80:, :, 2] > 170) & (image[80:, :, 1] < 100) & (image[80:, :, 0] < 100)).sum())
    assert red(frames[entry - 3]) < 10 and red(frames[entry + 3]) > 50


def test_clip_starts_before_the_entry_and_has_no_drawing(tmp_path, trackers):
    result = run(tmp_path, "red")
    entry = json.loads(result["clips_manifest"].read_text())["clips"][0]
    assert entry["trigger_frame_in_clip"] > 0 and entry["vehicle_boxes"][0]["frame_in_clip"] == 0
    first = next(iter_frames(result["clips_manifest"].parent / entry["clip_file"]))["image"]
    assert np.abs(first.astype(float) - head_frame("red").astype(float)).mean() < 3


def test_missing_scene_fails_before_loading_the_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(red_light, "load_detector", lambda: pytest.fail("the detector was loaded"))
    video = write_video(tmp_path / "small.mp4", [head_frame("red")] * 3)
    with pytest.raises(FileNotFoundError):
        detect_red_light_video(video, tmp_path / "out", tmp_path / "missing.json")


def test_read_frame_signals_reads_each_group():
    assert read_frame_signals(head_frame("red"), clip_config()) == {"through": "red", "left": "red"}


def test_group_with_no_signal_is_unknown():
    clip = clip_config()
    clip["signals"]["left"] = []
    assert read_frame_signals(head_frame("red"), clip)["left"] == "unknown"


def test_track_and_read_clip_covers_the_range(tmp_path, trackers):
    video = write_video(tmp_path / "small.mp4", [head_frame("red")] * 30)
    tracks, states = track_and_read_clip(video, clip_config(5, 25), ScriptedDetector(car_going_up))
    assert [row[0] for row in tracks["1"]] == list(range(5, 25))
    assert sorted(states["through"]) == list(range(5, 25)) and set(states["through"].values()) == {"red"}


def test_as_track_uses_the_majority_class():
    rows = [[1, 0, 0, 5, 5, 2, 0.9], [2, 0, 0, 5, 5, 7, 0.9], [3, 0, 0, 5, 5, 2, 0.9]]
    track = as_track("4", rows)
    assert track["class_name"] == "car" and track["frames"] == [1, 2, 3] and track["boxes"][0] == [0, 0, 5, 5]
