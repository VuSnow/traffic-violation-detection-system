import numpy as np
import pytest
import yaml
from conftest import ROOT
from notebook_data import OUTPUTS, history_from_rows, load_json, wrong_way_scene

from tvd.config import ConfigError
from tvd.domain import Box, TrackedObject
from tvd.motion import CameraMotion
from tvd.rules import RULES, Observation
from tvd.rules.wrong_way import WrongWayRule
from tvd.tracking.history import TrackHistory

SETTINGS = yaml.safe_load((ROOT / "configs" / "tvd" / "rules.yaml").read_text())["wrong_way"]
# One region over the whole 640x360 image, where traffic may only go up the image (towards a far vanishing point).
SCENE = {"vanishing_point": [320, -100000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360], [0, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}
STORED_TRACKS = OUTPUTS / "03_multi_object_tracking" / "tracks" / "video-test.ocsort-byte.jsonl"
STORED_MOTION = OUTPUTS / "04_traffic_scene_configuration" / "video-test.camera_motion.json"
needs_stored_tracks = pytest.mark.skipif(not (STORED_TRACKS.is_file() and STORED_MOTION.is_file()), reason="Needs the outputs of Notebooks 03 and 04, as on the GPU server.")


def judge(frames: int, dy: float, top_dy: float | None = None, scene: dict = SCENE, settings: dict = SETTINGS):
    """One car of 40x30 px, moving down the image by ``dy`` px per frame (its top edge by ``top_dy``), at 24 FPS.

    Going up, it starts low in the image, so that it stays away from the border.
    """
    history = TrackHistory()
    start = 130 if dy >= 0 else 330
    for k in range(frames):
        y2 = start + dy * k
        y1 = start - 30 + (dy if top_dy is None else top_dy) * k
        history.add(k, [TrackedObject(7, Box(300, y1, 340, y2), 2, "car", 0.9)])
    return WrongWayRule(scene, settings).evaluate(Observation(history.tracks, CameraMotion(), 24.0, 640, 360, 0, frames))


def test_the_rule_is_registered():
    assert "wrong_way" in RULES.names()


def test_moving_against_the_direction_for_the_whole_window_is_a_candidate():
    result = judge(60, dy=3)
    (finding,) = result.findings
    assert result.records == [{"track_id": "7", "verdict": "candidate", "evaluable_windows": 12}]
    assert (finding.start_frame, finding.trigger_frame, finding.end_frame) == (0, 48, 59) and finding.evidence["region_id"] == "road"
    assert finding.why == "Moved against the allowed direction of road by 4.80 box heights in about 2.0 s, 0 degrees from the opposite direction."


def test_moving_with_the_direction_is_no_violation():
    assert judge(60, dy=-3).verdict_counts() == {"no_violation": 1}


def test_a_track_shorter_than_the_window_cannot_be_judged():
    assert judge(30, dy=3).verdict_counts() == {"not_evaluable": 1}


def test_a_box_that_only_grows_is_no_violation():
    assert judge(60, dy=3, top_dy=0).verdict_counts() == {"no_violation": 1}  # C4: the top edge stays still.


def test_a_track_outside_every_region_is_not_judged():
    elsewhere = {**SCENE, "regions": [{**SCENE["regions"][0], "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]}]}
    assert judge(60, dy=3, scene=elsewhere).verdict_counts() == {"not_in_directional_region": 1}


def test_the_window_comes_from_the_settings():
    # 60 frames at 24 FPS are 2.5 s: enough for a 2 s window, not for a 3 s one.
    assert judge(60, dy=3, settings={**SETTINGS, "window_seconds": 3.0}).verdict_counts() == {"not_evaluable": 1}


def test_a_wrong_setting_is_named():
    with pytest.raises(ConfigError, match="settings.window_seconds: Input should be greater than 0"):
        WrongWayRule(SCENE, {**SETTINGS, "window_seconds": 0})


@needs_stored_tracks
@pytest.mark.parametrize("reverse", [False, True])  # Played backwards, every vehicle goes the wrong way.
def test_the_same_results_as_the_notebooks(reverse):
    import json

    from traffic_violation_detection import wrong_way as notebook
    from traffic_violation_detection.scene import load_camera_motion
    from traffic_violation_detection.tracking import add_to_tracks
    from traffic_violation_detection.video import read_video_metadata

    tracks: dict[str, list] = {}
    for line in STORED_TRACKS.read_text().splitlines():
        record = json.loads(line)
        add_to_tracks(tracks, record["frame_index"], record["tracks"])
    params = load_camera_motion(STORED_MOTION)
    if reverse:
        last = len(params) - 1
        tracks = {t: [[last - r[0], *r[1:]] for r in reversed(rows)] for t, rows in tracks.items()}
        params = params[::-1].copy()
    fps = read_video_metadata(ROOT / "data" / "video-test.mp4")["fps"]
    scene = load_json(ROOT / "configs" / "scenes" / "video-test.json")
    expected = {t: notebook.evaluate_track(notebook.build_track(t, rows, params, 1920, 1080), scene, notebook.rule_for_fps(fps), "video-test", fps) for t, rows in tracks.items()}

    rule = WrongWayRule(wrong_way_scene(scene), SETTINGS)
    result = rule.evaluate(Observation(history_from_rows(tracks).tracks, CameraMotion(params), fps, 1920, 1080, 0, len(params)))

    assert {r["track_id"]: (r["verdict"], r["evaluable_windows"]) for r in result.records} == {t: (e["status"], e["evaluable_windows"]) for t, e in expected.items()}
    events = sorted((e for r in expected.values() for e in r["events"]), key=lambda e: (e["trigger_frame"], e["track_id"]))
    assert len(events) > 0 or not reverse
    assert sorted(([f.track_id, f.class_name, f.start_frame, f.trigger_frame, f.end_frame, f.why, f.evidence] for f in result.findings), key=lambda f: (f[3], f[0])) == [
        [e["track_id"], e["class_name"], e["start_frame"], e["trigger_frame"], e["end_frame"], e["why"], {"region_id": e["region_id"], **e["evidence"]}] for e in events
    ]
