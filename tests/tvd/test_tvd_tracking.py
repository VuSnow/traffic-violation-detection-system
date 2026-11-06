import importlib.util

import numpy as np
import pytest

from tvd.domain import Box, Detection, Frame
from tvd.tracking import TRACKERS, BaseTracker
from tvd.tracking.boxmot_trackers import OcSortTracker

needs_boxmot = pytest.mark.skipif(importlib.util.find_spec("boxmot") is None, reason="Needs BoxMOT, as on the GPU server.")
IMAGE = np.zeros((240, 320, 3), np.uint8)


def moving_car(index: int) -> list[Detection]:
    return [Detection(Box(10 + 5 * index, 50, 50 + 5 * index, 80), 2, "car", 0.9)]


def test_a_tracker_must_implement_update_and_reset():
    class UpdateOnly(BaseTracker):
        def update(self, detections, frame):
            return []

    with pytest.raises(TypeError):
        UpdateOnly()


def test_the_boxmot_trackers_are_registered():
    assert {"bytetrack", "ocsort", "botsort"} <= set(TRACKERS.names())


@needs_boxmot
@pytest.mark.parametrize("name, options", [("bytetrack", {}), ("ocsort", {"use_byte": True}), ("botsort", {"use_cmc": False, "use_embeddings": False})])
def test_a_moving_car_keeps_one_track_id(name, options):
    tracker = TRACKERS.create(name, **options)
    seen = [tracked for index in range(12) for tracked in tracker.update(moving_car(index), Frame(index, index / 30, IMAGE))]
    assert seen and {t.track_id for t in seen} == {seen[0].track_id}
    assert all(t.class_name == "car" and t.class_id == 2 for t in seen)


@needs_boxmot
def test_no_detections_give_no_tracked_objects():
    assert TRACKERS.create("ocsort").update([], Frame(0, 0.0, IMAGE)) == []


@needs_boxmot
def test_reset_builds_a_new_tracker():
    class CountingOcSort(OcSortTracker):
        built = 0

        def _create(self, boxmot):
            CountingOcSort.built += 1
            return super()._create(boxmot)

    tracker = CountingOcSort(use_byte=True)
    tracker.reset()
    assert CountingOcSort.built == 2
