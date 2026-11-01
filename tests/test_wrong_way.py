"""Tests for traffic_violation_detection.wrong_way: the scenarios of Notebook 05, and the function from video to results."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fakes import PassThroughTracker, ScriptedDetector, write_video

from traffic_violation_detection import tracking, wrong_way
from traffic_violation_detection.scene import load_scene_config, region_by_id
from traffic_violation_detection.video import iter_frames, read_video_metadata
from traffic_violation_detection.wrong_way import CANDIDATE, NO_VIOLATION, NOT_EVALUABLE, NOT_IN_REGION, RULE, detect_wrong_way_video, evaluate_track, rule_for_fps

SCENE_PATH = Path(__file__).resolve().parents[1] / "configs" / "scenes" / "video-test.json"
FPS = 59.94
SCENE = load_scene_config(SCENE_PATH, 1920, 1080)
NO_DIRECTION = copy.deepcopy(SCENE)
region_by_id(NO_DIRECTION, "right_carriageway").update(allowed_direction=None, direction_tolerance_deg=None)


def synthetic_track(start, velocity, n=180, height=80, frame_step=1, top=None, clipped=False, seed=None):
    frames = np.arange(0, n, frame_step)
    bottom = np.array(start, dtype=np.float64) + frames[:, None] * np.array(velocity, dtype=np.float64)
    if seed is not None:
        bottom = bottom + np.random.default_rng(seed).normal(0, 3, bottom.shape)
    top_points = bottom - [0, height] if top is None else np.tile(np.array(top, dtype=np.float64), (len(frames), 1))
    return {
        "track_id": "1", "class_name": "car", "frames": frames, "bottom_ref": bottom, "top_ref": top_points,
        "heights": bottom[:, 1] - top_points[:, 1], "clipped": np.full(len(frames), clipped),
    }


DOWN, UP = (0.0, 1.0), (0.0, -1.0)
ALONG_LEFT_ROAD = (0.3, -0.45)
SCENARIOS = [
    ("correct direction, right carriageway", synthetic_track((1100, 500), UP), NO_VIOLATION, SCENE),
    ("wrong way, right carriageway", synthetic_track((1100, 330), DOWN), CANDIDATE, SCENE),
    ("correct direction, left carriageway", synthetic_track((780, 270), (0, 0.6), height=60), NO_VIOLATION, SCENE),
    ("wrong way, left carriageway", synthetic_track((760, 365), ALONG_LEFT_ROAD, height=40), CANDIDATE, SCENE),
    ("wrong way, far and small", synthetic_track((1010, 280), (0, 0.5), height=25), CANDIDATE, SCENE),
    ("creeping backward, under a box height", synthetic_track((1100, 400), (0, 0.3)), NO_VIOLATION, SCENE),
    ("stationary with position noise", synthetic_track((1100, 420), (0, 0), seed=0), NO_VIOLATION, SCENE),
    ("crossing the carriageway sideways", synthetic_track((960, 420), (0.6, 0.0)), NO_VIOLATION, SCENE),
    ("box only grows: bottom moves, top stays", synthetic_track((1100, 400), DOWN, n=130, top=(1100, 380)), NO_VIOLATION, SCENE),
    ("wrong way for less than the window", synthetic_track((1100, 400), DOWN, n=60), NOT_EVALUABLE, SCENE),
    ("wrong way but half the frames missing", synthetic_track((1100, 330), DOWN, frame_step=2), NOT_EVALUABLE, SCENE),
    ("wrong way but box clipped by the border", synthetic_track((1100, 330), DOWN, clipped=True), NOT_EVALUABLE, SCENE),
    ("wrong way inside the junction", synthetic_track((900, 700), DOWN), NOT_IN_REGION, SCENE),
    ("wrong way in the parking area", synthetic_track((1600, 380), DOWN, n=150), NOT_IN_REGION, SCENE),
    ("same path, but the scene lost its direction", synthetic_track((1100, 330), DOWN), NOT_IN_REGION, NO_DIRECTION),
]


@pytest.mark.parametrize("name, track, expected, scene", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_scenarios_of_notebook_05(name, track, expected, scene):
    result = evaluate_track(track, scene, rule_for_fps(FPS), "video-test", FPS)
    assert result["status"] == expected
    assert bool(result["events"]) == (expected == CANDIDATE)


def test_rule_matches_notebook_05_at_its_frame_rate():
    rule = rule_for_fps(FPS)
    assert (rule["window_frames"], rule["smoothing_frames"], rule["merge_gap_frames"]) == (120, 9, 15)


def test_rule_keeps_its_durations_at_another_frame_rate():
    rule = rule_for_fps(30.0)
    assert rule["window_frames"] == 60 and rule["smoothing_frames"] % 2 == 1 and rule["merge_gap_frames"] == 8
    assert RULE["window_seconds"] == rule["window_seconds"]


def test_event_is_pending_and_explains_itself():
    event = evaluate_track(SCENARIOS[1][1], SCENE, rule_for_fps(FPS), "video-test", FPS)["events"][0]
    assert event["review_status"] == "pending" and event["license_plate"] is None
    assert event["event_id"].startswith("evt-video-test-wrong_way-") and "right_carriageway" in event["why"]
    assert event["start_frame"] <= event["trigger_frame"] <= event["end_frame"]
    json.dumps(event)


WIDTH, HEIGHT = 480, 360


def small_scene(tmp_path):
    scene = {
        "schema_version": 1,
        "scene_id": "small",
        "video": {"width": WIDTH, "height": HEIGHT},
        "coordinates": {"reference_frame_index": 45, "always_visible_xyxy": [0, 0, WIDTH, HEIGHT]},
        "vanishing_point": [240, -300],
        "regions": [{"id": "road", "kind": "carriageway", "polygon": [[60, 10], [420, 10], [420, 350], [60, 350]], "allowed_direction": "away_from_camera", "direction_tolerance_deg": 60}],
        "lines": [],
        "signals": [],
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    return path


def small_video(tmp_path, n=90):
    noise = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    return write_video(tmp_path / "small.mp4", [cv2.GaussianBlur(noise, (0, 0), 1.2)] * n)


def moving_car(speed):
    # A car in the road region: positive speed moves down the image, against "away from the camera".
    return lambda call: [(200, 120 + speed * call, 280, 180 + speed * call, 0.9, 2)]


@pytest.fixture
def fake_tracker(monkeypatch):
    monkeypatch.setattr(tracking, "create_tracker", PassThroughTracker)


def test_video_with_a_wrong_way_car_gives_every_output(tmp_path, fake_tracker):
    video, scene = small_video(tmp_path), small_scene(tmp_path)
    result = detect_wrong_way_video(video, tmp_path / "out", scene, model=ScriptedDetector(moving_car(1.6)))
    assert len(result["events"]) == 1 and result["track_statuses"] == {"1": CANDIDATE}
    assert result["output_video"] == tmp_path / "out" / "small_output.mp4"
    assert sum(1 for _ in iter_frames(result["output_video"])) == 90
    manifest = json.loads(result["clips_manifest"].read_text())
    assert len(manifest["clips"]) == 1 and (result["clips_manifest"].parent / manifest["clips"][0]["clip_file"]).is_file()
    saved = json.loads(result["events_file"].read_text())
    assert saved["status_counts"] == {CANDIDATE: 1} and saved["candidates"][0]["review_status"] == "pending"


def test_video_with_a_correct_car_gives_no_event_and_an_empty_manifest(tmp_path, fake_tracker):
    video, scene = small_video(tmp_path), small_scene(tmp_path)
    result = detect_wrong_way_video(video, tmp_path / "out", scene, model=ScriptedDetector(lambda call: [(200, 280 - 1.6 * call, 280, 340 - 1.6 * call, 0.9, 2)]))
    assert result["events"] == [] and result["track_statuses"] == {"1": NO_VIOLATION}
    assert json.loads(result["clips_manifest"].read_text())["clips"] == []
    assert result["output_video"].is_file()


def test_result_video_marks_the_candidate_during_its_event_only(tmp_path, fake_tracker):
    video, scene = small_video(tmp_path), small_scene(tmp_path)
    result = detect_wrong_way_video(video, tmp_path / "out", scene, model=ScriptedDetector(moving_car(1.6)))
    frames = [r["image"] for r in iter_frames(result["output_video"])]
    event = result["events"][0]
    red = lambda image: int(((image[..., 2] > 180) & (image[..., 1] < 90) & (image[..., 0] < 90)).sum())
    assert red(frames[event["trigger_frame"]]) > 100
    assert event["start_frame"] == 0 or red(frames[event["start_frame"] - 1]) < 30


def test_clip_of_the_event_keeps_original_pixels(tmp_path, fake_tracker):
    video, scene = small_video(tmp_path), small_scene(tmp_path)
    result = detect_wrong_way_video(video, tmp_path / "out", scene, model=ScriptedDetector(moving_car(1.6)))
    entry = json.loads(result["clips_manifest"].read_text())["clips"][0]
    clip = next(iter_frames(result["clips_manifest"].parent / entry["clip_file"]))["image"]
    source = next(iter_frames(video))["image"]
    assert np.abs(clip.astype(float) - source.astype(float)).mean() < 12


def test_scene_of_another_size_is_rejected_before_loading_the_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(wrong_way, "load_detector", lambda: pytest.fail("the detector was loaded"))
    video = write_video(tmp_path / "other.mp4", [np.zeros((100, 200, 3), np.uint8)] * 3)
    with pytest.raises(ValueError, match="video is 200x100"):
        detect_wrong_way_video(video, tmp_path / "out", small_scene(tmp_path))
    assert read_video_metadata(video)["width"] == 200
