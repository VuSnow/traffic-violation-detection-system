"""Tests for the function from a video to red-light results, on a small drawn video with stand-in detector and tracker."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fakes import PassThroughTracker, ScriptedDetector, write_video

from traffic_violation_detection import red_light, tracking
from traffic_violation_detection.red_light import (
    AT_BORDER,
    CANDIDATE,
    COMPLIANT,
    UNKNOWN_SIGNAL,
    as_track,
    detect_red_light_video,
    evaluate_track,
    load_red_light_scene,
    read_frame_signals,
    to_reference_box,
    track_and_read_clip,
)
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
    tracks, states, detections = track_and_read_clip(video, clip_config(5, 25), ScriptedDetector(car_going_up))
    assert [row[0] for row in tracks["1"]] == list(range(5, 25)) and len(detections) == 20
    assert sorted(states["through"]) == list(range(5, 25)) and set(states["through"].values()) == {"red"}


def test_as_track_uses_the_majority_class():
    rows = [[1, 0, 0, 5, 5, 2, 0.9], [2, 0, 0, 5, 5, 7, 0.9], [3, 0, 0, 5, 5, 2, 0.9]]
    track = as_track("4", rows)
    assert track["class_name"] == "car" and track["frames"] == [1, 2, 3] and track["boxes"][0] == [0, 0, 5, 5]


LAMPS = {"red": [40, 30], "yellow": [40, 70], "green": [40, 110]}
LAMP_COLORS = {"red": (0, 0, 255), "yellow": (0, 230, 255), "green": (0, 255, 0)}
TOWARD_LANE = 90.0


def overlay_frame(lit):
    # A signal drawn over the image, with large lamps, as in a simulation.
    image = np.full((HEIGHT, WIDTH, 3), 40, np.uint8)
    cv2.circle(image, tuple(LAMPS[lit]), 16, LAMP_COLORS[lit], -1)
    return image


def toward_clip(**changes):
    return {
        "start": 0, "end": 60, "role": "simulation", "traffic_direction": "toward_camera", "entry_offset": 0.0,
        "stop_line": [[60, LINE_Y], [WIDTH, LINE_Y]], "lane_dividers": [60, WIDTH],
        "signals": {"method": "lamps", "lamp_half": 12, "through": [LAMPS], "left": [LAMPS]},
        **changes,
    }


def car_coming_closer(call):
    # Bottom edge from y=50 downward at 3 px per frame: it crosses the line near call 23, and goes well past it.
    bottom = 50 + 3 * call
    return [(140, bottom - 40, 180, bottom, 0.9, 2)] if bottom < HEIGHT - 8 else []


def write_simulation(tmp_path, lit, **scene_changes):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene_id": "small", "video": {"width": WIDTH, "height": HEIGHT}, "signal_source": "simulated_overlay",
        "lane_direction_deg": TOWARD_LANE, "clips": {"main": toward_clip()}, **scene_changes,
    }))
    return write_video(tmp_path / "small.mp4", [overlay_frame(lit)] * 60), path


def test_car_coming_toward_the_camera_on_red_is_a_candidate(tmp_path, trackers):
    video, scene = write_simulation(tmp_path, "red")
    result = detect_red_light_video(video, tmp_path / "out", scene, model=ScriptedDetector(car_coming_closer))
    assert [r["verdict"] for r in result["crossings"]["main"]] == [CANDIDATE]
    assert result["crossings"]["main"][0]["frame"] == pytest.approx(24, abs=3)


@pytest.mark.parametrize("lit", ["green", "yellow"])
def test_car_coming_toward_the_camera_on_another_signal_is_not_flagged(tmp_path, trackers, lit):
    video, scene = write_simulation(tmp_path, lit)
    result = detect_red_light_video(video, tmp_path / "out", scene, model=ScriptedDetector(car_coming_closer))
    assert result["events"] == [] and result["crossings"]["main"][0]["verdict"] == COMPLIANT


def test_car_driving_away_does_not_cross_in_a_scene_of_oncoming_traffic(tmp_path, trackers):
    video, scene = write_simulation(tmp_path, "red")
    result = detect_red_light_video(video, tmp_path / "out", scene, model=ScriptedDetector(car_going_up))
    assert result["crossings"]["main"] == []


def test_event_and_clip_carry_the_signal_source_of_the_scene(tmp_path, trackers):
    video, scene = write_simulation(tmp_path, "red")
    result = detect_red_light_video(video, tmp_path / "out", scene, model=ScriptedDetector(car_coming_closer))
    assert result["events"][0]["signal_source"] == "simulated_overlay" and result["events"][0]["review_status"] == "pending"
    assert json.loads(result["clips_manifest"].read_text())["clips"][0]["signal_source"] == "simulated_overlay"
    assert json.loads(result["events_file"].read_text())["signal_source"] == "simulated_overlay"


def test_lamps_drawn_large_are_read_with_a_larger_square():
    for lit in LAMPS:
        assert read_frame_signals(overlay_frame(lit), toward_clip())["through"] == lit


def test_moving_camera_keeps_the_line_fixed_on_the_road(tmp_path, trackers, monkeypatch):
    # The camera drifts so that each frame's content sits 1 px per frame lower than in the reference frame.
    params = np.array([[1.0, 0.0, 0.0, -float(k)] for k in range(60)])
    seen = {}

    def motion(video_path, detections, reference_frame, excluded_areas):
        seen.update(frames=len(detections), reference=reference_frame, areas=excluded_areas)
        return params

    monkeypatch.setattr(red_light, "estimate_camera_motion", motion)
    video, scene = write_simulation(tmp_path, "red", camera={"reference_frame": 7, "excluded_areas": [[0, 0, 80, 130]]})
    moved = detect_red_light_video(video, tmp_path / "moved", scene, model=ScriptedDetector(car_coming_closer))
    assert seen == {"frames": 60, "reference": 7, "areas": [[0, 0, 80, 130]]}
    # In road coordinates the car moves 2 px per frame, not 3, so it reaches the line later.
    assert moved["crossings"]["main"][0]["frame"] == pytest.approx(35, abs=3)


def test_stop_line_is_drawn_where_the_camera_shows_it(tmp_path, trackers, monkeypatch):
    params = np.array([[1.0, 0.0, 0.0, -20.0]] * 60)
    monkeypatch.setattr(red_light, "estimate_camera_motion", lambda *arguments: params)
    video, scene = write_simulation(tmp_path, "red", camera={"reference_frame": 0})
    result = detect_red_light_video(video, tmp_path / "out", scene, model=ScriptedDetector())
    frame = next(iter_frames(result["output_video"]))["image"]
    green = lambda row: int(((frame[row, :, 1] > 150) & (frame[row, :, 2] < 120)).sum())
    assert green(LINE_Y + 20) > 100 and green(LINE_Y) < 10


def test_border_is_judged_in_the_frame_and_not_on_the_road():
    frames = np.arange(40)
    boxes = np.c_[np.full(40, 100.0), frames * 5.0 - 40, np.full(40, 200.0), frames * 5.0 + 60]
    track = {"track_id": "1", "frames": frames, "boxes": boxes}
    states = {"through": {k: "red" for k in range(40)}, "left": {k: "red" for k in range(40)}}
    clip = toward_clip(stop_line=[[0, 120], [320, 120]], lane_dividers=[0, 320], end=40)
    assert evaluate_track(track, clip, states, TOWARD_LANE, WIDTH, HEIGHT)["verdict"] == CANDIDATE
    at_edge = {**track, "frame_boxes": np.c_[np.full(40, 0.0), boxes[:, 1:]]}
    assert evaluate_track(at_edge, clip, states, TOWARD_LANE, WIDTH, HEIGHT)["verdict"] == AT_BORDER


def test_reference_box_keeps_its_size_and_follows_the_camera():
    params = np.array([[1.0, 0.0, 10.0, -5.0]])
    assert to_reference_box([0, 100, 200, 140, 260], params) == pytest.approx([110.0, 195.0, 150.0, 255.0])


def test_as_track_keeps_frame_boxes_only_with_a_moving_camera():
    rows = [[0, 100, 200, 140, 260, 2, 0.9]]
    assert "frame_boxes" not in as_track("1", rows)
    moved = as_track("1", rows, np.array([[1.0, 0.0, 10.0, -5.0]]))
    assert moved["frame_boxes"] == [[100, 200, 140, 260]] and moved["boxes"][0] != moved["frame_boxes"][0]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"clips": {"main": toward_clip(traffic_direction="sideways")}}, "unknown traffic direction"),
        ({"camera": {"reference_frame": 0}, "clips": {"a": toward_clip(), "b": toward_clip()}}, "single clip"),
        ({"camera": {"reference_frame": 0}, "clips": {"main": toward_clip(start=5)}}, "single clip"),
    ],
)
def test_broken_simulation_scenes_are_rejected(tmp_path, changes, message):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"schema_version": 1, "video": {"width": WIDTH, "height": HEIGHT}, "lane_direction_deg": TOWARD_LANE, "clips": {"main": toward_clip()}, **changes}))
    with pytest.raises(ValueError, match=message):
        load_red_light_scene(path, WIDTH, HEIGHT)


def test_committed_simulation_scene_loads():
    path = Path(__file__).resolve().parents[1] / "configs" / "scenes" / "traffic_video_modified.json"
    scene = load_red_light_scene(path, 1920, 1012)
    clip = scene["clips"]["main"]
    assert scene["signal_source"] == "simulated_overlay" and clip["traffic_direction"] == "toward_camera" and clip["entry_offset"] == 0.0
    assert 0 <= scene["camera"]["reference_frame"] < clip["end"]
