import importlib.util

import pytest
from conftest import ROOT
from notebook_data import RED_LIGHT_SCENE, RLR_VIDEO, load_json, red_light_section

from tvd.camera import CameraConfig, load_rule_defaults
from tvd.check import check_video
from tvd.config import ComponentConfig, ConfigError
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
from tvd.video import VideoReader

DEFAULTS = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
# The test car comes down the image by 1 px per frame: towards the camera, against the only allowed direction.
CAMERA = {
    "camera_id": "test-camera",
    "frame_size": [640, 360],
    "rules": {
        "red_light": {
            "scene": {"stop_line": [[0, 150], [640, 150]], "lane_dividers": [0, 600], "lane_direction_deg": 90, "traffic_direction": "toward_camera",
                      "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}},
            "settings": {"entry_offset": 0.0, "pass_margin": 0.2},
        },
        "wrong_way": {"scene": {"vanishing_point": [320, -100000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360], [0, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}},
    },
}
needs_gpu_and_rlr = pytest.mark.skipif(
    any(importlib.util.find_spec(n) is None for n in ("ultralytics", "boxmot")) or not RLR_VIDEO.is_file() or not (ROOT / "weights" / "yolo26m.pt").is_file(),
    reason="Needs Ultralytics, BoxMOT, the weights and data/raw/rlr_occasions.mp4, as on the GPU server.",
)


def run(sample_video, camera: dict = CAMERA, start: int = 0):
    config = CameraConfig.model_validate(camera)
    with VideoReader(sample_video) as reader:
        return check_video(
            reader, config, build_detector(ComponentConfig(type="test-moving-car", options={"step": 0, "step_y": 1})),
            build_tracker(ComponentConfig(type="test-same-id")), {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS), start,
        )


def test_one_pass_gives_every_rule_the_same_tracks(sample_video):
    result = run(sample_video)
    assert (result.frames, result.tracks, list(result.results)) == (240, 1, ["red_light", "wrong_way"])
    (red,), (wrong,) = result.results["red_light"].findings, result.results["wrong_way"].findings
    assert (red.trigger_frame, red.evidence["signal_state"]) == (70, "red")
    assert (wrong.start_frame, wrong.trigger_frame, wrong.end_frame) == (0, 48, 239)


def test_the_result_can_be_written_as_json(sample_video):
    import json

    data = json.loads(json.dumps(run(sample_video).as_json()))
    assert data["rules"]["red_light"]["verdicts"] == {"candidate": 1} and data["rules"]["wrong_way"]["settings"]["window_seconds"] == 2.0


def test_a_video_of_another_size_is_refused(sample_video):
    with pytest.raises(ConfigError, match="Camera test-camera has 1920x1080 frames, but traffic_sample.mp4 is 640x360"):
        run(sample_video, {**CAMERA, "frame_size": [1920, 1080]})


@needs_gpu_and_rlr
def test_the_same_crossings_as_the_notebooks_from_the_video(monkeypatch):
    from traffic_violation_detection import red_light as notebook
    from traffic_violation_detection.detection import load_detector

    monkeypatch.chdir(ROOT)
    scene, clip_name = load_json(RED_LIGHT_SCENE), "C"
    clip = scene["clips"][clip_name]
    tracks, states, _ = notebook.track_and_read_clip(RLR_VIDEO, clip, load_detector())
    expected = notebook.evaluate_clip(notebook.join_clip_tracks(tracks, None), clip, states, scene["lane_direction_deg"], 1280, 720, 30.0)

    camera = CameraConfig.model_validate({"camera_id": "rlr-C", "frame_size": [1280, 720], "rules": {"red_light": red_light_section(scene, clip_name)}})
    perception = {"weights": "yolo26m.pt", "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]}
    with VideoReader(RLR_VIDEO) as reader:
        result = check_video(
            reader, camera, build_detector(ComponentConfig(type="yolo", options=perception)), build_tracker(ComponentConfig(type="ocsort", options={"use_byte": True})),
            {"bicycle", "car", "motorcycle", "bus", "truck"}, build_motion_estimator(camera.motion), build_rules(camera, DEFAULTS), clip["start"], clip["end"],
        )
    assert result.results["red_light"].records == expected and len(result.results["red_light"].findings) == 2


def test_the_json_leaves_out_the_points_of_each_finding(sample_video):
    findings = [f for rule in run(sample_video).as_json()["rules"].values() for f in rule["findings"]]
    assert len(findings) == 2 and all("points" not in f for f in findings)


def test_the_first_frame_analysed_is_kept(sample_video):
    result = run(sample_video, start=100)
    assert (result.first_frame, result.frames, result.as_json()["first_frame"]) == (100, 140, 100)


def test_more_observers_see_every_frame_of_the_pass(sample_video):
    from tvd.observers import FrameCounter

    reported = []
    config = CameraConfig.model_validate(CAMERA)
    with VideoReader(sample_video) as reader:
        check_video(
            reader, config, build_detector(ComponentConfig(type="test-moving-car", options={"step": 0, "step_y": 1})), build_tracker(ComponentConfig(type="test-same-id")),
            {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS), 40, observers=[FrameCounter(reported.append, every=50, total=200)],
        )
    assert reported == [50, 100, 150, 200]
