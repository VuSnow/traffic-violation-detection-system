import pytest

from tvd.motion import CameraMotion
from tvd.rules import BaseRule, Finding, Observation, RuleResult


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
