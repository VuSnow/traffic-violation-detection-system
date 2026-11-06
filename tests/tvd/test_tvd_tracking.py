import pytest

from tvd.tracking import BaseTracker


def test_a_tracker_must_implement_update_and_reset():
    class UpdateOnly(BaseTracker):
        def update(self, detections, frame):
            return []

    with pytest.raises(TypeError):
        UpdateOnly()
