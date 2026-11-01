"""License plate reading, developed in Notebook 06. Its results support human review and never decide a case.

A plate is read from several frames of a vehicle and the readings are compared, because one frame's
confidence is not enough: the OCR can be sure of a wrong text, and of an empty one. Each vehicle
ends in one explicit status:

- ``read``: enough frames agree on one text;
- ``uncertain``: there is a text, but too few frames agree, so it is only a suggestion;
- ``unreadable``: a plate was found, but no frame gave a usable text;
- ``no_plate_found``: no plate was found on this vehicle.

Small functions come first. ``read_plates_video`` and ``read_plates_from_clips`` at the end join them:
one goes from any video to a copy showing each vehicle's plate, and the other from the clips of
candidate violations to a plate result for each event.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from traffic_violation_detection.detection import OUTPUT_CRF, load_detector
from traffic_violation_detection.tracking import collect_tracks, majority_class
from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer

PLATE_SETTINGS = {
    "version": 1,
    "detector_model": "yolo-v9-t-640-license-plate-end2end",
    "ocr_model": "cct-s-v2-global-model",
    "cpu_threads": 8,
    "frames_per_track": 12,    # The frames where the vehicle looks largest.
    "min_frame_gap": 4,        # Neighbouring frames are nearly the same, so they would not be independent votes.
    "vehicle_margin": 0.15,    # A box can cut off the plate, so the crop extends beyond it.
    "lowest_plate_center": 0.4,  # A plate's center is below this fraction of the box height: this rejects roof signs.
    "min_plate_width": 30,     # px; narrower plates are too small to read.
    "min_chars": 4,
    "min_char_prob": 0.95,     # The least sure character of a usable read. A wrong plate is worse than none, so this is strict.
    "min_votes": 2,            # Frames that must agree for a ``read``.
    "min_agreement": 0.67,     # The share of usable frames that must agree: 2 of 3 is not enough, and 3 of 4 is.
}

READ = "read"
UNCERTAIN = "uncertain"
UNREADABLE = "unreadable"
NO_PLATE = "no_plate_found"

PAD_CHAR = "_"
STATUS_COLORS_BGR = {READ: (80, 220, 80), UNCERTAIN: (0, 220, 255), UNREADABLE: (170, 170, 170), NO_PLATE: (235, 235, 235)}


def load_plate_models(settings: dict = PLATE_SETTINGS) -> tuple:
    """The plate detector and the OCR, on the CPU on purpose.

    Running ONNX on the GPU moved PyTorch's later detections by about 0.01 px, enough to change 15 of
    136 tracks, so results depended on what had run before. On the CPU the detections stay identical,
    and the plate results were the same on both (Notebook 06).
    """
    try:
        import onnxruntime as ort
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector
    except ImportError as error:
        raise RuntimeError('Plate reading needs its models: pip install -e ".[plates]"') from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings["cpu_threads"]
    options.inter_op_num_threads = 1
    arguments = {"providers": ["CPUExecutionProvider"], "sess_options": options}
    return create_detector(settings["detector_model"], **arguments), LicensePlateRecognizer(settings["ocr_model"], **arguments)


def select_frames(rows: list, count: int, min_gap: int) -> list:
    """The rows where the vehicle looks largest, at least ``min_gap`` frames apart. Rows start with ``frame, x1, y1, x2, y2``."""
    chosen: list = []
    for row in sorted(rows, key=lambda r: r[4] - r[2], reverse=True):
        if all(abs(row[0] - c[0]) >= min_gap for c in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def crop_vehicle(image: np.ndarray, box, margin: float) -> tuple[np.ndarray, tuple[int, int]]:
    """The vehicle's region with a margin around its box, and the position of that region in the frame."""
    x1, y1, x2, y2 = box
    margin_x, margin_y = margin * (x2 - x1), margin * (y2 - y1)
    height, width = image.shape[:2]
    left, top = max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))
    right, bottom = min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))
    return image[top:bottom, left:right], (left, top)


def pick_vehicle_plate(plates: list, box, offset: tuple[int, int], lowest_center: float):
    """The detected plate most likely to be this vehicle's: inside its box, and nearest the middle of its bottom edge.

    A box can also hold a neighbour's plate, a roof sign or a shop sign, so the most confident detection is not enough.
    """
    x1, y1, x2, y2 = box
    width, height = max(x2 - x1, 1), max(y2 - y1, 1)
    best, best_distance = None, float("inf")
    for plate in plates:
        b = plate.bounding_box
        across = ((b.x1 + b.x2) / 2 + offset[0] - x1) / width
        down = ((b.y1 + b.y2) / 2 + offset[1] - y1) / height
        if not (0 <= across <= 1 and lowest_center <= down <= 1.05):
            continue
        distance = float(np.hypot(across - 0.5, down - 1.0))
        if distance < best_distance:
            best, best_distance = plate, distance
    return best


def read_plate_text(reader, plate_bgr: np.ndarray) -> dict:
    """OCR of one plate crop. Confidence covers real characters only, because the OCR is also sure of its padding."""
    # The OCR assumes an array is already RGB; only image files get converted by it.
    prediction = reader.run_one(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB), return_confidence=True, remove_pad_char=False)
    raw = prediction.plate
    probs = np.asarray(prediction.char_probs, dtype=np.float64)
    real = [i for i, char in enumerate(raw) if char != PAD_CHAR]
    return {
        "text": "".join(raw[i] for i in real),
        "mean_prob": round(float(probs[real].mean()), 4) if real else 0.0,
        "min_prob": round(float(probs[real].min()), 4) if real else 0.0,
        "has_gap": PAD_CHAR in raw.rstrip(PAD_CHAR),
    }


def rejection_reason(read: dict, settings: dict = PLATE_SETTINGS) -> str | None:
    """Why a single read cannot be used, or None if it can."""
    if read["plate_width"] < settings["min_plate_width"]:
        return "too_small"
    if len(read["text"]) < settings["min_chars"]:
        return "too_short"
    if read["has_gap"]:
        return "gap_in_text"
    if read["min_prob"] < settings["min_char_prob"]:
        return "low_confidence"
    return None


def read_vehicle_in_frame(image: np.ndarray, row: list, detector, reader, settings: dict = PLATE_SETTINGS) -> dict | None:
    """Read one vehicle's plate in one frame. None means no plate of this vehicle was found there."""
    box = row[1:5]
    crop, offset = crop_vehicle(image, box, settings["vehicle_margin"])
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    plate = pick_vehicle_plate(detector.predict(crop), box, offset, settings["lowest_plate_center"])
    if plate is None:
        return None
    b = plate.bounding_box
    plate_image = crop[max(0, int(b.y1)):max(0, int(b.y2)), max(0, int(b.x1)):max(0, int(b.x2))]
    if plate_image.size == 0:
        return None
    read = read_plate_text(reader, plate_image)
    read.update(
        frame=int(row[0]),
        detection_confidence=round(float(plate.confidence), 4),
        plate_width=int(plate_image.shape[1]),
        plate_height=int(plate_image.shape[0]),
        plate_box_xyxy=[int(b.x1) + offset[0], int(b.y1) + offset[1], int(b.x2) + offset[0], int(b.y2) + offset[1]],
        plate_image=plate_image.copy(),
    )
    read["rejected"] = rejection_reason(read, settings)
    return read


def combine_reads(reads: list[dict], settings: dict = PLATE_SETTINGS) -> dict:
    """One result from all the frames of a vehicle: its status, its text if any, and how far the frames agree."""
    usable = [r for r in reads if r["rejected"] is None]
    result = {"status": NO_PLATE, "text": None, "votes": 0, "usable_reads": len(usable), "plates_found": len(reads), "agreement": 0.0, "mean_prob": 0.0, "other_texts": {}}
    if not reads:
        return result
    if not usable:
        return {**result, "status": UNREADABLE}
    votes = Counter(r["text"] for r in usable)
    # On a tie, the text with more total confidence wins.
    text = max(votes, key=lambda t: (votes[t], sum(r["mean_prob"] for r in usable if r["text"] == t)))
    agreement = votes[text] / len(usable)
    agreed = votes[text] >= settings["min_votes"] and agreement >= settings["min_agreement"]
    return {
        **result,
        "status": READ if agreed else UNCERTAIN,
        "text": text,
        "votes": votes[text],
        "agreement": round(agreement, 3),
        "mean_prob": round(float(np.mean([r["mean_prob"] for r in usable if r["text"] == text])), 4),
        "other_texts": {t: n for t, n in votes.items() if t != text},
    }


def best_read(reads: list[dict], text: str | None) -> dict | None:
    """The read to keep as evidence: of the chosen text if there is one, and with the largest plate."""
    pool = [r for r in reads if r["text"] == text and r["rejected"] is None] or reads
    return max(pool, key=lambda r: r["plate_width"] * r["plate_height"], default=None)


def read_tracks(video_path: str | Path, tracks: dict[str, list], detector, reader, settings: dict = PLATE_SETTINGS) -> dict[str, dict]:
    """Read the plate of every track. Returns each one's result, with its reads for evidence and analysis."""
    wanted: dict[int, list] = {}
    for track_id, rows in tracks.items():
        for row in select_frames(rows, settings["frames_per_track"], settings["min_frame_gap"]):
            wanted.setdefault(int(row[0]), []).append((track_id, row))
    reads: dict[str, list] = {track_id: [] for track_id in tracks}
    if wanted:
        for record in iter_frames(video_path, min(wanted), max(wanted) + 1):
            for track_id, row in wanted.get(record["frame_index"], []):
                read = read_vehicle_in_frame(record["image"], row, detector, reader, settings)
                if read is not None:
                    reads[track_id].append(read)
    return {track_id: {**combine_reads(r, settings), "reads": r} for track_id, r in reads.items()}


def save_plate_crop(result: dict, folder: Path, name: str) -> str | None:
    """Save a result's evidence crop without further compression, and return its file name."""
    read = best_read(result["reads"], result["text"])
    if read is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    file_name = f"{name}_f{read['frame']:05d}.png"
    cv2.imwrite(str(folder / file_name), read["plate_image"])
    return file_name


def public_result(result: dict) -> dict:
    """A result without its images, ready to be saved as JSON."""
    reads = [{k: v for k, v in r.items() if k != "plate_image"} for r in result["reads"]]
    return {**{k: v for k, v in result.items() if k != "reads"}, "reads": reads}


def plate_label(result: dict) -> str:
    """The text shown for a vehicle: a question mark says the reading is only a suggestion."""
    if result["status"] == READ:
        return result["text"]
    if result["status"] == UNCERTAIN:
        return f"{result['text']}?"
    return "unreadable" if result["status"] == UNREADABLE else "no plate"


def draw_plate_result(image: np.ndarray, box, result: dict, label_prefix: str = "") -> None:
    """Draw a vehicle's box and its plate result on a BGR frame, in place."""
    color = STATUS_COLORS_BGR[result["status"]]
    thickness = max(1, round(image.shape[0] / 540))
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    scale = max(0.5, 0.8 * image.shape[0] / 1080)
    text = f"{label_prefix}{plate_label(result)}"
    font_thickness = max(1, round(scale * 2.5))
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, font_thickness)
    top = max(height + 8, y1)
    cv2.rectangle(image, (x1, top - height - 8), (x1 + width + 6, top), color, -1)
    cv2.putText(image, text, (x1 + 3, top - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


def rows_by_frame(tracks: dict[str, list]) -> dict[int, list[tuple[str, list]]]:
    """For each frame, the tracks present in it and their rows."""
    frames: dict[int, list] = {}
    for track_id, rows in tracks.items():
        for row in rows:
            frames.setdefault(int(row[0]), []).append((track_id, row))
    return frames


def write_plate_video(video_path: Path, output_path: Path, fps: float, tracks: dict, results: dict) -> None:
    """Write a copy of the video showing each vehicle's box and plate result."""
    present = rows_by_frame(tracks)
    with video_writer(output_path, fps, OUTPUT_CRF) as write_frame:
        for record in iter_frames(video_path):
            for track_id, row in present.get(record["frame_index"], []):
                draw_plate_result(record["image"], row[1:5], results[track_id])
            write_frame(record["image"])


def read_plates_video(video_path: str | Path, output_folder_path: str | Path, model=None, plate_models: tuple | None = None, settings: dict = PLATE_SETTINGS) -> dict:
    """Track the vehicles of any video and read each one's plate.

    In ``output_folder_path`` this saves:

    - ``<video name>_output.mp4``: a copy showing each vehicle's box and plate result;
    - ``<video name>_plate_crops/``: the plate crop kept as evidence for each vehicle;
    - ``<video name>_plates.json``: each vehicle's status, text, and the reads behind them.

    Returns these paths with the results. Earlier files of these names are replaced.
    """
    video_path, folder = Path(video_path), Path(output_folder_path)
    fps = read_video_metadata(video_path)["fps"]
    detector = model if model is not None else load_detector()
    plate_detector, reader = plate_models if plate_models is not None else load_plate_models(settings)
    video_id = video_path.stem

    tracks, _ = collect_tracks(video_path, detector)
    results = read_tracks(video_path, tracks, plate_detector, reader, settings)

    output_video = output_video_path(video_path, folder)
    write_plate_video(video_path, output_video, fps, tracks, results)
    crops = folder / f"{video_id}_plate_crops"
    saved = {}
    for track_id, result in results.items():
        crop = save_plate_crop(result, crops, f"t{track_id}")
        saved[track_id] = {**public_result(result), "class_name": majority_class(tracks[track_id]), "plate_crop": crop}
    plates_file = folder / f"{video_id}_plates.json"
    plates_file.write_text(json.dumps({"video_id": video_id, "settings": settings, "status_counts": dict(Counter(r["status"] for r in results.values())), "tracks": saved}, indent=1))
    return {"output_video": output_video, "plates_file": plates_file, "crops_folder": crops, "results": results, "tracks": tracks}


def read_plates_from_clips(manifest_path: str | Path, output_folder_path: str | Path, plate_models: tuple | None = None, settings: dict = PLATE_SETTINGS) -> dict:
    """Read the plate of each candidate violation, from the clips and ``manifest.json`` of a violation step.

    In ``output_folder_path`` this saves one ``<event id>_output.mp4`` showing the vehicle and its plate
    result, ``plate_crops/``, and ``plates.json`` with a ``license_plate`` result for each event.
    A clip's vehicle is known from the manifest, so no detector or tracker runs here.
    """
    manifest_path, folder = Path(manifest_path), Path(output_folder_path)
    manifest = json.loads(manifest_path.read_text())
    plate_detector, reader = plate_models if plate_models is not None else load_plate_models(settings)
    folder.mkdir(parents=True, exist_ok=True)

    events, results = [], {}
    for clip in manifest["clips"]:
        event_id, clip_path = clip["event_id"], manifest_path.parent / clip["clip_file"]
        rows = [[b["frame_in_clip"], *b["box_xyxy"]] for b in clip["vehicle_boxes"]]
        result = read_tracks(clip_path, {event_id: rows}, plate_detector, reader, settings)[event_id]
        results[event_id] = result
        write_plate_video(clip_path, folder / f"{event_id}_output.mp4", clip["fps"], {event_id: rows}, {event_id: result})
        events.append({
            "event_id": event_id,
            "clip_file": clip["clip_file"],
            "license_plate": public_result(result),
            "plate_crop": save_plate_crop(result, folder / "plate_crops", event_id),
            "review_status": clip["review_status"],
        })
    plates_file = folder / "plates.json"
    plates_file.write_text(json.dumps({"video_id": manifest["video_id"], "settings": settings, "status_counts": dict(Counter(r["status"] for r in results.values())), "events": events}, indent=1))
    return {"plates_file": plates_file, "events": events, "results": results}


def edit_distance(a: str, b: str) -> int:
    """Insertions, deletions and substitutions that turn one text into the other."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def score_plate_results(results: dict[str, dict], labels: dict[str, dict]) -> dict:
    """Compare plate results with labels, per labelled track.

    With a readable label, a track ends ``correct_read``, ``wrong_read``, ``correct_suggestion``,
    ``wrong_suggestion``, ``declined`` (unreadable) or ``no_plate``. With a label saying the plate cannot
    be read by eye, a ``read`` or ``uncertain`` status is an ``invented_read`` or ``invented_suggestion``,
    because nobody could confirm that text. Labels marked ``unsure`` are only counted.
    """
    outcome: Counter = Counter()
    distances = []
    for track_id, label in labels.items():
        result = results.get(track_id, {"status": NO_PLATE, "text": None})
        status, text = result["status"], result["text"]
        if label.get("unsure"):
            outcome["unsure_label"] += 1
        elif "text" in label:
            outcome["readable_labels"] += 1
            if status == READ:
                outcome["correct_read" if text == label["text"] else "wrong_read"] += 1
            elif status == UNCERTAIN:
                outcome["correct_suggestion" if text == label["text"] else "wrong_suggestion"] += 1
            else:
                outcome["declined" if status == UNREADABLE else "no_plate"] += 1
            if status in (READ, UNCERTAIN) and text != label["text"]:
                distances.append(edit_distance(text, label["text"]))
        else:
            outcome["unreadable_labels"] += 1
            if status == READ:
                outcome["invented_read"] += 1
            elif status == UNCERTAIN:
                outcome["invented_suggestion"] += 1
    return {**{k: outcome[k] for k in ("readable_labels", "correct_read", "wrong_read", "correct_suggestion", "wrong_suggestion", "declined", "no_plate", "unreadable_labels", "invented_read", "invented_suggestion", "unsure_label")}, "wrong_text_edit_distances": sorted(distances)}


def mask_plate(text: str | None) -> str | None:
    """A text safe to show in a committed notebook: the first two characters, and its length."""
    return None if text is None else text[:2] + "*" * (len(text) - 2)
