import numpy as np
import pytest
import yaml
from conftest import ROOT
from notebook_data import OUTPUTS, RED_LIGHT_SCENE, history_from_rows, load_json, red_light_section

from tvd.config import ConfigError
from tvd.domain import Box, Frame, TrackedObject
from tvd.motion import CameraMotion
from tvd.rules import RULES, Observation
from tvd.rules.red_light import RedLightRule
from tvd.signals import SignalTimeline
from tvd.tracking.history import TrackHistory

SETTINGS = yaml.safe_load((ROOT / "configs" / "tvd" / "rules.yaml").read_text())["red_light"]
VERSION_1 = {**SETTINGS, "version": 1, "join_fragments": False, "estimate_hidden_entry": False}
# A horizontal stop line at y = 200 on a 640x360 image. Traffic goes up the image, away from the camera.
SCENE = {
    "stop_line": [[0, 200], [640, 200]], "lane_dividers": [100, 540], "lane_direction_deg": -90, "traffic_direction": "away_from_camera",
    "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}], "left": []},
}
STORED = OUTPUTS / "05b_red_light_violation"
needs_stored_clips = pytest.mark.skipif(not (STORED / "tracks.json").is_file(), reason="Needs the outputs of Notebook 05b, as on the GPU server.")


def car_up(frames: int = 60, x: float = 300, stop_at: float | None = None, turn_right_after: int | None = None) -> list[tuple[float, float]]:
    """Bottom-left corners of a 40x30 px car going up the image by 4 px per frame."""
    path = []
    for k in range(frames):
        y2 = 330 - 4 * k if stop_at is None else max(330 - 4 * k, stop_at)
        dx = 0
        if turn_right_after is not None and k > turn_right_after:
            y2, dx = 330 - 4 * turn_right_after, 6 * (k - turn_right_after)
        path.append((x + dx, y2))
    return path


def judge(path, state: str = "red", settings: dict = SETTINGS, scene: dict = SCENE):
    history = TrackHistory()
    for k, (x1, y2) in enumerate(path):
        history.add(k, [TrackedObject(7, Box(x1, y2 - 30, x1 + 40, y2), 2, "car", 0.9)])
    signals = SignalTimeline({"through": {k: state for k in range(len(path))}, "left": {k: "unknown" for k in range(len(path))}})
    return RedLightRule(scene, settings).judge(Observation(history.tracks, CameraMotion(), 24.0, 640, 360, 0, len(path)), signals)


def test_the_rule_is_registered():
    assert "red_light" in RULES.names()


def test_entering_on_red_and_passing_the_line_is_a_candidate():
    result = judge(car_up())
    assert result.records == [{
        "track_id": "7", "class_name": "car", "frame": 31, "x": 320.0, "lane": "through", "signal_group": "through", "signal_state": "red",
        "state_for_frames": 31, "movement": "straight", "passed_line": True, "entry_seen": True, "entry_estimated": False, "verdict": "candidate",
        "box_xyxy": [300.0, 176.0, 340.0, 206.0],
    }]
    (finding,) = result.findings
    assert (finding.start_frame, finding.trigger_frame, finding.end_frame) == (0, 31, 59)
    assert finding.why == "Entered in lane 'through' while the through signal was red (1.3 s after it was first seen in that state), and continued past the stop line."


@pytest.mark.parametrize("state, verdict", [("green", "signal_not_red"), ("yellow", "signal_not_red"), ("unknown", "not_evaluable_signal"), ("conflict", "not_evaluable_signal")])
def test_the_signal_decides(state, verdict):
    assert judge(car_up(), state).verdict_counts() == {verdict: 1}


def test_a_vehicle_that_stops_on_the_line_is_not_a_candidate():
    assert judge(car_up(80, stop_at=206)).verdict_counts() == {"stopped_at_line": 1}


def test_a_right_turn_on_red_is_not_a_candidate():
    assert judge(car_up(60, turn_right_after=35)).verdict_counts() == {"right_turn_on_red": 1}


def test_a_box_at_the_image_border_cannot_be_judged():
    assert judge(car_up(x=598)).verdict_counts() == {"not_evaluable_border": 1}


def test_the_left_turn_lane_follows_the_left_signal():
    assert judge(car_up(x=30)).records[0]["signal_group"] == "left"  # The left group has no heads here: its state is unknown.


def test_a_vehicle_that_never_crosses_has_no_record():
    assert judge(car_up(20)).records == []


def test_the_rule_reads_the_signals_itself_during_the_pass():
    rule = RedLightRule(SCENE, SETTINGS)
    (monitor,) = rule.frame_observers()
    history = TrackHistory()
    for k, (x1, y2) in enumerate(car_up()):
        monitor.observe(Frame(k, k / 24, np.zeros((360, 640, 3), np.uint8)), [])
        history.add(k, [TrackedObject(7, Box(x1, y2 - 30, x1 + 40, y2), 2, "car", 0.9)])
    assert rule.evaluate(Observation(history.tracks, CameraMotion(), 24.0, 640, 360, 0, 60)).verdict_counts() == {"candidate": 1}


@pytest.mark.parametrize("change, message", [
    ({"settings": {"pass_margin": -1}}, "settings.pass_margin: Input should be greater than or equal to 0"),
    ({"scene": {"stop_line": [[640, 200], [0, 200]]}}, "scene.stop_line: Value error, x must increase from left to right"),
    ({"scene": {"signals": {"reader": "laser", "through": [{}]}}}, "Unknown signal reader 'laser'"),
])
def test_a_wrong_setting_is_named(change, message):
    with pytest.raises(ValueError, match=message):  # ConfigError is a ValueError.
        RedLightRule({**SCENE, **change.get("scene", {})}, {**SETTINGS, **change.get("settings", {})})


def notebook_inputs(clip_name: str):
    scene, stored_tracks, stored_signals = load_json(RED_LIGHT_SCENE), load_json(STORED / "tracks.json"), load_json(STORED / "signals.json")
    states = {group: {int(k): v for k, v in by_frame.items()} for group, by_frame in stored_signals[clip_name].items()}
    return scene, scene["clips"][clip_name], stored_tracks[clip_name], states


@needs_stored_clips
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("moving", [False, True], ids=["fixed camera", "moving camera"])
def test_the_same_results_as_the_notebooks_on_every_clip(version, moving):
    from traffic_violation_detection import red_light as notebook

    rule_v = notebook.RULE if version == 2 else notebook.RULE_V1
    settings = SETTINGS if version == 2 else VERSION_1
    for clip_name in load_json(RED_LIGHT_SCENE)["clips"]:
        scene, clip, tracks, states = notebook_inputs(clip_name)
        # A made-up camera shake, so that the moving-camera code is compared too.
        params = np.array([[1.0, 0.0005 * np.sin(k / 40), 3 * np.sin(k / 25), 2 * np.cos(k / 35)] for k in range(clip["end"] + 1)]) if moving else None
        joined = notebook.join_clip_tracks(tracks, params, rule_v)
        expected = notebook.evaluate_clip(joined, clip, states, scene["lane_direction_deg"], 1280, 720, 30.0, rule_v, params)

        section = red_light_section(scene, clip_name)
        rule = RedLightRule(section["scene"], {**settings, **section["settings"]})
        motion = CameraMotion() if params is None else CameraMotion(params)
        observation = Observation(history_from_rows(tracks).tracks, motion, 30.0, 1280, 720, clip["start"], clip["end"])
        result = rule.judge(observation, SignalTimeline(states))

        assert result.records == expected, clip_name
        events = [notebook.build_event(r, clip_name, "rlr", 30.0, clip["start"], "camera", rule_v) for r in expected if r["verdict"] == "candidate"]
        assert [(f.track_id, f.class_name, f.trigger_frame, f.why, f.evidence) for f in result.findings] == [
            (e["track_id"], e["class_name"], e["trigger_frame"], e["why"], {**e["evidence"], "signal_source": "camera"}) for e in events
        ], clip_name
