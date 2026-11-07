import pytest

from tvd.domain import Box
from tvd.motion import CameraMotion
from tvd.rules import BaseRule, Finding, Observation, RuleResult
from tvd.tracking.history import TrackPoint


def test_a_rule_must_implement_evaluate():
    class Incomplete(BaseRule):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_a_rule_needs_no_frames_unless_it_says_so():
    class Counting(BaseRule):
        name = "counting"

        def evaluate(self, observation):
            return RuleResult([], [{"track_id": str(t), "verdict": "seen"} for t in observation.tracks], {})

    rule = Counting()
    result = rule.evaluate(Observation({1: None, 2: None}, CameraMotion(), 30.0, 640, 360, 0, 10))
    assert rule.frame_observers() == [] and result.verdict_counts() == {"seen": 2}


def test_a_finding_cannot_be_changed():
    finding = Finding("wrong_way", "7", "car", 10, 20, 30, "why")
    with pytest.raises(AttributeError):
        finding.trigger_frame = 25


def test_a_finding_keeps_its_vehicle_but_leaves_it_out_of_its_json():
    points = (TrackPoint(10, Box(0, 0, 10, 10), 0.9, "car"), TrackPoint(11, Box(1, 0, 11, 10), 0.9, "car"))
    finding = Finding("wrong_way", "7", "car", 10, 11, 11, "why", {"region_id": "road"}, points)
    assert finding.points[1].box == Box(1, 0, 11, 10)
    assert finding.as_json() == {
        "rule": "wrong_way", "track_id": "7", "class_name": "car", "start_frame": 10, "trigger_frame": 11, "end_frame": 11, "why": "why", "evidence": {"region_id": "road"},
    }
