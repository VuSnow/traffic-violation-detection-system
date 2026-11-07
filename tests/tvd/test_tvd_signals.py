import numpy as np
import pytest
from notebook_data import RED_LIGHT_SCENE, RLR_VIDEO, load_json, red_light_section

from tvd.domain import Frame
from tvd.signals import SIGNAL_READERS, SignalMonitor, combine_states, smooth_states
from tvd.video import VideoReader

needs_rlr_video = pytest.mark.skipif(not RLR_VIDEO.is_file(), reason="Needs data/raw/rlr_occasions.mp4, which only the GPU server has.")
LAMPS = {"red": [20, 10], "yellow": [20, 30], "green": [20, 50]}
BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 255, 0)}


def image_with(colour: str | None, size: int = 6) -> np.ndarray:
    """A dark 64x64 image, with one lamp lit."""
    image = np.zeros((64, 64, 3), np.uint8)
    if colour:
        x, y = LAMPS[colour]
        image[y - size:y + size + 1, x - size:x + size + 1] = BGR[colour]
    return image


@pytest.mark.parametrize("colour", ["red", "yellow", "green"])
def test_lamps_read_the_lit_one(colour):
    assert SIGNAL_READERS.create("lamps", **LAMPS).read(image_with(colour)) == colour


def test_lamps_all_dark_are_unknown():
    assert SIGNAL_READERS.create("lamps", **LAMPS).read(image_with(None)) == "unknown"


def test_a_lamp_too_close_to_the_border_counts_nothing():
    assert SIGNAL_READERS.create("lamps", red=[2, 10], yellow=[20, 30], green=[20, 50]).read(image_with(None)) == "unknown"


def test_glow_of_a_grey_image_is_unknown():
    assert SIGNAL_READERS.create("glow", roi=[0, 0, 64, 64]).read(np.full((64, 64, 3), 200, np.uint8)) == "unknown"


def test_glow_reads_its_colour():
    assert SIGNAL_READERS.create("glow", roi=[0, 0, 64, 64]).read(image_with("green", size=10)) == "green"


def test_glow_with_an_empty_box_is_refused():
    with pytest.raises(ValueError, match="is empty"):
        SIGNAL_READERS.create("glow", roi=[10, 10, 10, 20])


def test_glow_outside_the_image_is_unknown():
    assert SIGNAL_READERS.create("glow", roi=[100, 100, 120, 120]).read(image_with("green")) == "unknown"


@pytest.mark.parametrize("states, expected", [
    (["red", "red"], "red"), (["red", "unknown"], "red"), (["unknown", "unknown"], "unknown"), (["red", "green"], "conflict"), ([], "unknown"),
])
def test_heads_of_one_group_combine(states, expected):
    assert combine_states(states) == expected


def test_smoothing_removes_single_frame_flicker():
    assert smooth_states(["red"] * 4 + ["green"] + ["red"] * 4) == ["red"] * 9


def test_a_monitor_smooths_each_group_and_leaves_a_group_without_heads_unknown():
    monitor = SignalMonitor({"through": [SIGNAL_READERS.create("test-fixed-state", state="green")], "left": []}, smoothing_frames=3)
    for index in range(5, 8):
        monitor.observe(Frame(index, 0.0, np.zeros((2, 2, 3), np.uint8)), [])
    timeline = monitor.timeline()
    assert timeline.as_dict() == {"through": {5: "green", 6: "green", 7: "green"}, "left": {5: "unknown", 6: "unknown", 7: "unknown"}}
    assert timeline.get("through", 8) is None and timeline.state("through", 8) == "unknown"


@needs_rlr_video
@pytest.mark.parametrize("clip_name", ["A", "C", "H"])  # Day lamps, night glow, infrared.
def test_the_same_states_as_the_notebooks(clip_name):
    from traffic_violation_detection.red_light import read_frame_signals, smooth_states as notebook_smooth

    scene = load_json(RED_LIGHT_SCENE)
    clip = scene["clips"][clip_name]
    signals = red_light_section(scene, clip_name)["scene"]["signals"]
    groups = {g: [SIGNAL_READERS.create(signals["reader"], **signals["options"], **head) for head in signals[g]] for g in ("through", "left")}
    monitor = SignalMonitor(groups)
    raw = {"through": [], "left": []}
    with VideoReader(RLR_VIDEO) as reader:
        for frame in reader.frames(clip["start"], clip["start"] + 90):
            monitor.observe(frame, [])
            for group, state in read_frame_signals(frame.image, clip).items():
                raw[group].append(state)
    expected = {group: {clip["start"] + i: s for i, s in enumerate(notebook_smooth(states))} for group, states in raw.items()}
    assert monitor.timeline().as_dict() == expected
