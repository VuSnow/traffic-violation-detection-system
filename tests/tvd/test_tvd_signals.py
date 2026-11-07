import numpy as np
import pytest

from tvd.signals import SIGNAL_READERS, combine_states, smooth_states

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


@pytest.mark.parametrize("states, expected", [
    (["red", "red"], "red"), (["red", "unknown"], "red"), (["unknown", "unknown"], "unknown"), (["red", "green"], "conflict"), ([], "unknown"),
])
def test_heads_of_one_group_combine(states, expected):
    assert combine_states(states) == expected


def test_smoothing_removes_single_frame_flicker():
    assert smooth_states(["red"] * 4 + ["green"] + ["red"] * 4) == ["red"] * 9
