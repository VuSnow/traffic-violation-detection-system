"""Tests for traffic_violation_detection.evaluation, on small hand-made labels."""

import importlib.util

import numpy as np
import pytest

from traffic_violation_detection.evaluation import (
    apply_review_verdicts,
    average_precision,
    box_iou,
    detection_counts,
    load_mot_ground_truth,
    match_boxes,
    read_review_verdicts,
    tracking_scores,
    wilson_interval,
)

needs_motmetrics = pytest.mark.skipif(importlib.util.find_spec("motmetrics") is None, reason="motmetrics is not installed here.")
A, B, C = [0, 0, 10, 10], [100, 100, 120, 120], [200, 0, 230, 30]


def det(box, confidence=0.9):
    return {"bbox_xyxy": box, "confidence": confidence}


def test_ground_truth_rows_become_boxes_on_video_frames(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,7,10,20,30,40,1,-1,-1,-1\n1,8,0,0,5,5,1,-1,-1,-1\n3,7,12,20,30,40,1,-1,-1,-1\n")
    truth = load_mot_ground_truth(path, frame_offset=-1)
    assert truth == {0: [(7, [10.0, 20.0, 40.0, 60.0]), (8, [0.0, 0.0, 5.0, 5.0])], 2: [(7, [12.0, 20.0, 42.0, 60.0])]}


def test_a_single_row_is_read(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("0,1,0,0,10,10,1,-1,-1,-1\n")
    assert load_mot_ground_truth(path) == {0: [(1, [0.0, 0.0, 10.0, 10.0])]}


def test_iou_of_known_boxes():
    iou = box_iou([A, A], [A, [5, 0, 15, 10], B])
    assert iou.shape == (2, 3)
    assert iou[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_iou_with_no_boxes_is_empty():
    assert box_iou([], [A]).shape == (0, 1) and box_iou([A], []).shape == (1, 0)


def test_matching_is_one_to_one():
    assert match_boxes([A, A], [A], 0.5) == [(0, 0, 1.0)] or match_boxes([A, A], [A], 0.5) == [(1, 0, 1.0)]
    assert len(match_boxes([A, A], [A], 0.5)) == 1


def test_matching_keeps_only_pairs_above_the_threshold():
    assert match_boxes([A], [[5, 0, 15, 10]], 0.5) == [] and len(match_boxes([A], [[5, 0, 15, 10]], 0.3)) == 1


def test_matching_finds_the_best_total():
    # Greedy matching would give the first box to the first label and leave the second label unmatched.
    truth = [[0, 0, 10, 10], [6, 0, 16, 10]]
    boxes = [[3, 0, 13, 10], [8, 0, 18, 10]]
    assert sorted((i, j) for i, j, _ in match_boxes(truth, boxes, 0.3)) == [(0, 0), (1, 1)]


def test_detection_counts():
    truth = {0: [(1, A), (2, B)], 1: [(1, A)]}
    detections = {0: [det(A), det(C)], 1: [], 2: [det(A)]}
    counts = detection_counts(truth, detections)
    assert (counts["true_positives"], counts["false_positives"], counts["misses"]) == (1, 1, 2)
    assert counts["precision"] == 0.5 and counts["recall"] == pytest.approx(1 / 3) and counts["mean_iou"] == 1.0


def test_low_confidence_detections_can_be_left_out():
    truth = {0: [(1, A)]}
    counts = detection_counts(truth, {0: [det(A, 0.2), det(C, 0.1)]}, min_confidence=0.25)
    assert (counts["true_positives"], counts["false_positives"], counts["misses"]) == (0, 0, 1) and counts["precision"] is None


def test_perfect_detections_have_full_average_precision():
    truth = {0: [(1, A), (2, B)], 1: [(1, C)]}
    assert average_precision(truth, {0: [det(A), det(B)], 1: [det(C)]}) == pytest.approx(1.0)


def test_a_false_positive_ranked_first_lowers_average_precision():
    truth = {0: [(1, A)]}
    # Precision is 0 at the first detection and 0.5 when the label is found, so the area is 0.5.
    assert average_precision(truth, {0: [det(C, 0.9), det(A, 0.5)]}) == pytest.approx(0.5)


def test_a_second_box_on_the_same_object_is_a_false_positive():
    truth = {0: [(1, A), (2, B)]}
    # Found: A (1/1), A again (1/2), B (2/3). Recall 0.5 at precision 1, then 1.0 at precision 2/3.
    assert average_precision(truth, {0: [det(A, 0.9), det(A, 0.8), det(B, 0.7)]}) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_no_detections_give_zero_average_precision():
    assert average_precision({0: [(1, A)]}, {}) == 0.0


def test_average_precision_needs_labels():
    with pytest.raises(ValueError):
        average_precision({}, {0: [det(A)]})


@needs_motmetrics
def test_perfect_tracks_score_fully():
    truth = {k: [(1, A), (2, B)] for k in range(5)}
    tracks = {k: [(10, A), (20, B)] for k in range(5)}
    scores = tracking_scores(truth, tracks)
    assert scores["mota"] == 1.0 and scores["idf1"] == 1.0 and scores["num_switches"] == 0 and scores["motp"] == pytest.approx(1.0)
    assert scores["mostly_tracked"] == 2 and scores["num_unique_objects"] == 2


@needs_motmetrics
def test_a_track_that_changes_id_counts_one_switch():
    truth = {k: [(1, A)] for k in range(6)}
    tracks = {k: [(10 if k < 3 else 11, A)] for k in range(6)}
    scores = tracking_scores(truth, tracks)
    assert scores["num_switches"] == 1 and scores["idf1"] == pytest.approx(0.5)


@needs_motmetrics
def test_frames_without_a_track_are_misses():
    truth = {k: [(1, A)] for k in range(4)}
    scores = tracking_scores(truth, {0: [(10, A)], 1: [(10, A)]})
    assert scores["num_misses"] == 2 and scores["recall"] == 0.5 and scores["num_false_positives"] == 0


@needs_motmetrics
def test_a_track_too_far_from_the_label_is_a_miss_and_a_false_positive():
    scores = tracking_scores({0: [(1, A)]}, {0: [(10, [5, 0, 15, 10])]})
    assert scores["num_misses"] == 1 and scores["num_false_positives"] == 1


def test_wilson_interval_matches_notebook_05b():
    low, high = wilson_interval(13, 13)
    assert (round(low, 2), high) == (0.77, 1.0)
    low, high = wilson_interval(13, 14)
    assert (round(low, 2), round(high, 2)) == (0.69, 0.99)
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


SHEET = "order,case_id,review_clip,clip_role,ai_label,rule_result,violation_clip,ai_note,your_verdict,your_note\n"
LABELS = {
    "clips": {"A": {"events": [{"id": "A1", "label": "violation"}, {"id": "A-n1", "label": "not_violation"}]}},
    "development_clips": {"H": {"events": [{"id": "H1", "label": "violation"}]}},
}


def test_blank_verdicts_are_left_out(tmp_path):
    path = tmp_path / "sheet.csv"
    path.write_text(SHEET + "1,A1,01_A1.mp4,evaluation,violation,candidate,,note,,\n2,H1,02_H1.mp4,development,violation,x,,note, Not_Violation ,stopped\n")
    assert read_review_verdicts(path) == {"H1": "not_violation"}


def test_an_unknown_verdict_is_refused(tmp_path):
    path = tmp_path / "sheet.csv"
    path.write_text(SHEET + "1,A1,01_A1.mp4,evaluation,violation,candidate,,note,yes,\n")
    with pytest.raises(ValueError, match="A1"):
        read_review_verdicts(path)


def test_reviewed_verdicts_replace_labels_in_a_copy():
    reviewed = apply_review_verdicts(LABELS, {"A1": "uncertain"})
    event = reviewed["clips"]["A"]["events"][0]
    assert event["label"] == "uncertain" and event["ai_label"] == "violation" and event["reviewed_by_a_person"]
    assert LABELS["clips"]["A"]["events"][0] == {"id": "A1", "label": "violation"}
    assert "reviewed_by_a_person" not in reviewed["clips"]["A"]["events"][1]


def test_a_verdict_for_an_unknown_case_is_refused():
    with pytest.raises(ValueError, match="Z9"):
        apply_review_verdicts(LABELS, {"Z9": "violation"})
