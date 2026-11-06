"""Measurements against labels. Validated in Notebook 09.

Detection and tracking are compared with ground truth in the MOTChallenge text format, whose rows are
``frame, id, x, y, width, height, ...``. Those labels have no classes, so detections are counted per
object, whatever their class. Boxes are ``[x1, y1, x2, y2]`` pixels.

A person's verdicts from the review sheet of Notebook 05b, when given, replace the labels that the AI
assistant made.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REVIEW_VERDICTS = ("violation", "not_violation", "uncertain")
TRACKING_METRICS = (
    "mota", "motp", "idf1", "idp", "idr", "precision", "recall", "num_switches", "num_fragmentations",
    "mostly_tracked", "partially_tracked", "mostly_lost", "num_unique_objects", "num_false_positives", "num_misses",
)


def load_mot_ground_truth(path: str | Path, frame_offset: int = 0) -> dict[int, list[tuple[int, list[float]]]]:
    """``{video frame: [(object id, box), ...]}`` from a MOTChallenge text file.

    ``frame_offset`` is added to each frame number, so that it becomes the video's frame index: -1 for a
    file that counts frames from 1.
    """
    truth: dict[int, list[tuple[int, list[float]]]] = {}
    for row in np.loadtxt(path, delimiter=",", ndmin=2):
        frame, object_id, x, y, width, height = row[:6]
        truth.setdefault(int(frame) + frame_offset, []).append((int(object_id), [float(x), float(y), float(x + width), float(y + height)]))
    return truth


def box_iou(a, b) -> np.ndarray:
    """Intersection over union of every box in ``a`` with every box in ``b``, as an array of ``len(a)`` rows."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    width = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    height = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = width * height
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(truth_boxes, boxes, iou_threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one pairs ``(truth index, box index, IoU)``, each at least ``iou_threshold``, with the largest total overlap."""
    iou = box_iou(truth_boxes, boxes)
    if not iou.size:
        return []
    cost = np.where(iou >= iou_threshold, -iou, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if iou[i, j] >= iou_threshold]


def detection_counts(truth: dict, detections: dict[int, list[dict]], iou_threshold: float = 0.5, min_confidence: float = 0.0) -> dict:
    """True positives, false positives and misses over the labelled frames.

    ``detections`` maps a frame to dicts with ``bbox_xyxy`` and ``confidence``. Frames without labels are
    left out, because nothing there can be checked.
    """
    tp = fp = misses = 0
    ious: list[float] = []
    for frame, objects in truth.items():
        boxes = [d["bbox_xyxy"] for d in detections.get(frame, []) if d["confidence"] >= min_confidence]
        pairs = match_boxes([box for _, box in objects], boxes, iou_threshold)
        tp += len(pairs)
        fp += len(boxes) - len(pairs)
        misses += len(objects) - len(pairs)
        ious += [iou for _, _, iou in pairs]
    return {
        "true_positives": tp,
        "false_positives": fp,
        "misses": misses,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + misses) if tp + misses else None,
        "mean_iou": float(np.mean(ious)) if ious else None,
    }


def average_precision(truth: dict, detections: dict[int, list[dict]], iou_threshold: float = 0.5) -> float:
    """Area under the precision-recall curve, with all-point interpolation.

    Detections are ranked by confidence over all labelled frames. Each takes the unmatched label it
    overlaps most, if that overlap reaches ``iou_threshold``; otherwise it is a false positive.
    """
    total = sum(len(objects) for objects in truth.values())
    if not total:
        raise ValueError("There are no labelled objects")
    labelled = {frame: np.array([box for _, box in objects]).reshape(-1, 4) for frame, objects in truth.items()}
    taken = {frame: np.zeros(len(boxes), dtype=bool) for frame, boxes in labelled.items()}
    ranked = sorted(((d["confidence"], frame, d["bbox_xyxy"]) for frame in truth for d in detections.get(frame, [])), key=lambda r: -r[0])
    hits = []
    for _, frame, box in ranked:
        iou = box_iou(labelled[frame], [box])[:, 0]
        iou[taken[frame]] = -1.0
        best = int(np.argmax(iou)) if iou.size else -1
        hit = best >= 0 and iou[best] >= iou_threshold
        if hit:
            taken[frame][best] = True
        hits.append(hit)
    if not hits:
        return 0.0
    true_positives = np.cumsum(hits)
    recall = np.concatenate([[0.0], true_positives / total, [1.0]])
    precision = np.concatenate([[0.0], true_positives / np.arange(1, len(hits) + 1), [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.nonzero(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def tracking_scores(truth: dict, tracks: dict[int, list[tuple[int, list[float]]]], iou_threshold: float = 0.5) -> dict:
    """CLEAR MOT and identity scores, from motmetrics.

    ``tracks`` maps a frame to ``(track id, box)`` pairs. Frames without labels are left out. ``motp``
    is given as the mean IoU of matched pairs, so higher is better.
    """
    try:
        import motmetrics
    except ImportError as error:
        raise RuntimeError('Tracking scores need motmetrics: pip install -e ".[evaluation]"') from error
    accumulator = motmetrics.MOTAccumulator(auto_id=True)
    for frame in sorted(truth):
        objects = truth[frame]
        hypotheses = tracks.get(frame, [])
        distance = 1.0 - box_iou([box for _, box in objects], [box for _, box in hypotheses])
        distance[distance > 1.0 - iou_threshold] = np.nan
        accumulator.update([object_id for object_id, _ in objects], [track_id for track_id, _ in hypotheses], distance)
    summary = motmetrics.metrics.create().compute(accumulator, metrics=list(TRACKING_METRICS), name="run")
    scores = {name: summary[name].iloc[0].item() for name in TRACKING_METRICS}
    scores["motp"] = 1.0 - scores["motp"]
    return scores


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a proportion: honest for small samples, unlike a normal approximation."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def read_review_verdicts(sheet_path: str | Path) -> dict[str, str]:
    """A person's verdicts in the review sheet of Notebook 05b, by case id. Cases left blank are left out."""
    verdicts = {}
    with Path(sheet_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            verdict = (row.get("your_verdict") or "").strip().lower()
            if not verdict:
                continue
            if verdict not in REVIEW_VERDICTS:
                raise ValueError(f"Case {row['case_id']}: verdict {verdict!r} is not one of {REVIEW_VERDICTS}")
            verdicts[row["case_id"]] = verdict
    return verdicts


def apply_review_verdicts(labels: dict, verdicts: dict[str, str]) -> dict:
    """A copy of the red-light labels in which each reviewed case has the person's verdict, and says so."""
    reviewed = copy.deepcopy(labels)
    found = set()
    for group in ("clips", "development_clips"):
        for clip in reviewed.get(group, {}).values():
            for event in clip["events"]:
                if event["id"] in verdicts:
                    event["ai_label"] = event["label"]
                    event["label"] = verdicts[event["id"]]
                    event["reviewed_by_a_person"] = True
                    found.add(event["id"])
    unknown = set(verdicts) - found
    if unknown:
        raise ValueError(f"Reviewed cases not in the labels: {sorted(unknown)}")
    return reviewed
