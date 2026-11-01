"""Tests for traffic_violation_detection.plates, with stand-ins for the plate detector and the OCR."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fakes import PassThroughTracker, ScriptedDetector, write_video

from traffic_violation_detection import plates, tracking
from traffic_violation_detection.plates import (
    NO_PLATE,
    PLATE_SETTINGS,
    READ,
    UNCERTAIN,
    UNREADABLE,
    best_read,
    combine_reads,
    crop_vehicle,
    edit_distance,
    mask_plate,
    pick_vehicle_plate,
    plate_label,
    public_result,
    read_plate_text,
    read_plates_from_clips,
    read_plates_video,
    read_tracks,
    read_vehicle_in_frame,
    rejection_reason,
    score_plate_results,
    select_frames,
)
from traffic_violation_detection.video import iter_frames


def detection(x1, y1, x2, y2, confidence=0.9):
    return SimpleNamespace(confidence=confidence, bounding_box=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2))


def prediction(raw, prob=0.99, slots=10):
    raw = raw.ljust(slots, "_")
    return SimpleNamespace(plate=raw, char_probs=np.full(slots, prob))


class FakeReader:
    """Gives each call the next text of a script, and keeps the images it was shown."""

    def __init__(self, texts=("AB1234",), prob=0.99):
        self.texts = list(texts)
        self.prob = prob
        self.images = []

    def run_one(self, image, return_confidence=False, remove_pad_char=True):
        assert return_confidence and not remove_pad_char
        self.images.append(image.copy())
        return prediction(self.texts[(len(self.images) - 1) % len(self.texts)], self.prob)


class FakePlateDetector:
    """Finds one plate at a fixed place in each crop, as a share of the crop's size."""

    def __init__(self, area=(0.3, 0.7, 0.7, 0.9)):
        self.area = area

    def predict(self, crop):
        if self.area is None:
            return []
        height, width = crop.shape[:2]
        a, b, c, d = self.area
        return [detection(int(a * width), int(b * height), int(c * width), int(d * height))]


def read(text="AB1234", rejected=None, mean_prob=0.99, width=60, height=30, frame=0):
    return {"text": text, "rejected": rejected, "mean_prob": mean_prob, "plate_width": width, "plate_height": height, "frame": frame, "plate_image": np.zeros((height, width, 3), np.uint8)}


def row(frame, height=100, x=100, y=100, width=100):
    return [frame, x, y, x + width, y + height, 2, 0.9]


def test_frames_are_the_largest_views_and_spread_out():
    rows = [row(k, height=100 + k) for k in range(30)]
    assert [r[0] for r in select_frames(rows, 3, 4)] == [29, 25, 21]


def test_fewer_frames_than_asked_is_fine():
    assert len(select_frames([row(0), row(1)], 12, 4)) == 1


def test_crop_extends_beyond_the_box_and_reports_its_position():
    crop, offset = crop_vehicle(np.zeros((400, 600, 3), np.uint8), [100, 100, 200, 200], 0.15)
    assert offset == (85, 85) and crop.shape[:2] == (130, 130)


def test_crop_stops_at_the_image_border():
    crop, offset = crop_vehicle(np.zeros((400, 600, 3), np.uint8), [0, 0, 100, 390], 0.15)
    assert offset == (0, 0) and crop.shape[0] == 400


BOX = [100, 100, 300, 400]


def test_plate_nearest_the_bottom_middle_wins_over_a_more_confident_one():
    roof_like = detection(150, 250, 250, 280, confidence=0.99)
    bottom = detection(170, 360, 230, 390, confidence=0.6)
    assert pick_vehicle_plate([roof_like, bottom], BOX, (0, 0), 0.4) is bottom


def test_roof_sign_alone_is_not_taken_for_a_plate():
    assert pick_vehicle_plate([detection(150, 110, 250, 140)], BOX, (0, 0), 0.4) is None


def test_plate_of_a_neighbour_outside_the_box_is_ignored():
    assert pick_vehicle_plate([detection(320, 360, 380, 390)], BOX, (0, 0), 0.4) is None


def test_plate_position_accounts_for_the_crop_offset():
    plate = detection(70, 260, 130, 290)  # In crop coordinates: with this offset, near the bottom middle of the box.
    assert pick_vehicle_plate([plate], BOX, (100, 100), 0.4) is plate


def test_ocr_receives_rgb_not_bgr():
    reader = FakeReader()
    blue_bgr = np.zeros((20, 40, 3), np.uint8)
    blue_bgr[..., 0] = 255
    read_plate_text(reader, blue_bgr)
    assert reader.images[0][0, 0].tolist() == [0, 0, 255]


def test_confidence_ignores_the_padding():
    reader = SimpleNamespace(run_one=lambda *a, **k: SimpleNamespace(plate="AB12______", char_probs=np.array([0.5, 0.7, 0.9, 0.5, 1, 1, 1, 1, 1, 1.0])))
    result = read_plate_text(reader, np.zeros((20, 40, 3), np.uint8))
    assert result["text"] == "AB12" and result["mean_prob"] == pytest.approx(0.65) and result["min_prob"] == pytest.approx(0.5)


def test_empty_reading_has_no_confidence_even_if_the_ocr_is_sure_of_its_padding():
    result = read_plate_text(FakeReader([""], prob=1.0), np.zeros((20, 40, 3), np.uint8))
    assert result["text"] == "" and result["mean_prob"] == 0.0 and not result["has_gap"]


def test_gap_inside_the_text_is_noticed():
    result = read_plate_text(FakeReader(["AB_12"]), np.zeros((20, 40, 3), np.uint8))
    assert result["has_gap"] and result["text"] == "AB12"


def single(**changes):
    return {"text": "AB1234", "has_gap": False, "min_prob": 0.99, "plate_width": 60, **changes}


@pytest.mark.parametrize(
    "changes, reason",
    [({}, None), ({"plate_width": 20}, "too_small"), ({"text": "AB"}, "too_short"), ({"text": ""}, "too_short"), ({"has_gap": True}, "gap_in_text"), ({"min_prob": 0.1}, "low_confidence")],
)
def test_rejection_reasons(changes, reason):
    assert rejection_reason(single(**changes)) == reason


def test_no_reads_means_no_plate_found():
    assert combine_reads([])["status"] == NO_PLATE


def test_plate_with_only_unusable_reads_is_unreadable():
    result = combine_reads([read(rejected="too_small"), read(rejected="too_short")])
    assert result["status"] == UNREADABLE and result["text"] is None and result["plates_found"] == 2


def test_frames_that_agree_give_a_read():
    result = combine_reads([read(), read(), read(), read("AB1284")])
    assert result["status"] == READ and result["text"] == "AB1234" and result["votes"] == 3
    assert result["agreement"] == pytest.approx(0.75) and result["other_texts"] == {"AB1284": 1}


def test_two_frames_of_three_are_not_enough():
    result = combine_reads([read(), read(), read("AB1284")])
    assert result["status"] == UNCERTAIN and result["text"] == "AB1234"


def test_a_single_frame_is_never_enough_however_sure():
    result = combine_reads([read(mean_prob=1.0)])
    assert result["status"] == UNCERTAIN and result["text"] == "AB1234"


def test_low_agreement_gives_only_a_suggestion():
    result = combine_reads([read("AAAA11"), read("AAAA11"), read("BBBB22"), read("CCCC33"), read("DDDD44")])
    assert result["status"] == UNCERTAIN and result["agreement"] == pytest.approx(0.4)


def test_a_tie_goes_to_the_more_confident_text():
    result = combine_reads([read("AAAA11", mean_prob=0.6), read("BBBB22", mean_prob=0.9)])
    assert result["text"] == "BBBB22" and result["status"] == UNCERTAIN


def test_rejected_reads_do_not_vote():
    result = combine_reads([read(), read(), read("ZZZZ99", rejected="low_confidence"), read("ZZZZ99", rejected="low_confidence"), read("ZZZZ99", rejected="low_confidence")])
    assert result["text"] == "AB1234" and result["status"] == READ and result["usable_reads"] == 2


def test_best_read_is_the_largest_plate_of_the_chosen_text():
    reads = [read(width=50, frame=1), read(width=90, frame=2), read("OTHER1", width=200, frame=3)]
    assert best_read(reads, "AB1234")["frame"] == 2


def test_best_read_of_an_unreadable_plate_is_still_kept_as_evidence():
    assert best_read([read("", rejected="too_short", frame=7)], None)["frame"] == 7


def test_best_read_of_nothing_is_nothing():
    assert best_read([], None) is None


def test_public_result_has_no_images_and_is_json():
    result = {**combine_reads([read(), read()]), "reads": [read(), read()]}
    json.dumps(public_result(result))


@pytest.mark.parametrize(
    "status, text, label",
    [(READ, "AB1234", "AB1234"), (UNCERTAIN, "AB1234", "AB1234?"), (UNREADABLE, None, "unreadable"), (NO_PLATE, None, "no plate")],
)
def test_plate_label_shows_how_sure_the_result_is(status, text, label):
    assert plate_label({"status": status, "text": text}) == label


def test_frame_read_records_the_plate_in_frame_coordinates():
    image = np.full((400, 600, 3), 90, np.uint8)
    result = read_vehicle_in_frame(image, row(5, height=200, width=200, x=200, y=100), FakePlateDetector(), FakeReader())
    assert result["frame"] == 5 and result["text"] == "AB1234" and result["rejected"] is None
    x1, y1, x2, y2 = result["plate_box_xyxy"]
    assert 200 <= x1 < x2 <= 400 and 100 <= y1 < y2 <= 340
    assert result["plate_image"].shape[:2] == (y2 - y1, x2 - x1)


def test_frame_with_no_plate_gives_nothing():
    assert read_vehicle_in_frame(np.zeros((400, 600, 3), np.uint8), row(5, height=200, width=200), FakePlateDetector(None), FakeReader()) is None


def test_tiny_vehicle_is_skipped():
    assert read_vehicle_in_frame(np.zeros((400, 600, 3), np.uint8), row(5, height=6, width=6), FakePlateDetector(), FakeReader()) is None


W, H = 480, 360


def scene_video(tmp_path, name="small.mp4", n=40):
    noise = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    return write_video(tmp_path / name, [cv2.GaussianBlur(noise, (0, 0), 2)] * n)


def test_read_tracks_reads_each_track_from_its_own_frames(tmp_path):
    video = scene_video(tmp_path)
    tracks = {"1": [row(k, height=160, width=160) for k in range(40)], "2": [row(k, height=160, width=160, x=250) for k in range(20)]}
    results = read_tracks(video, tracks, FakePlateDetector(), FakeReader())
    assert results["1"]["status"] == READ and results["1"]["plates_found"] == 10
    assert results["2"]["plates_found"] == 5 and {r["frame"] % 4 for r in results["2"]["reads"]} == {0} or results["2"]["status"] == READ


def test_read_tracks_with_no_tracks_is_empty(tmp_path):
    assert read_tracks(scene_video(tmp_path, n=2), {}, FakePlateDetector(), FakeReader()) == {}


@pytest.fixture
def fake_tracker(monkeypatch):
    monkeypatch.setattr(tracking, "create_tracker", PassThroughTracker)


def car(call):
    return [(100, 100, 300, 300, 0.9, 2)]


def test_video_function_saves_every_output(tmp_path, fake_tracker):
    video = scene_video(tmp_path)
    result = read_plates_video(video, tmp_path / "out", model=ScriptedDetector(car), plate_models=(FakePlateDetector(), FakeReader()))
    assert result["output_video"] == tmp_path / "out" / "small_output.mp4"
    assert sum(1 for _ in iter_frames(result["output_video"])) == 40
    saved = json.loads(result["plates_file"].read_text())
    assert saved["tracks"]["1"]["status"] == READ and saved["tracks"]["1"]["text"] == "AB1234" and saved["status_counts"] == {READ: 1}
    assert (result["crops_folder"] / saved["tracks"]["1"]["plate_crop"]).is_file()


def test_video_with_no_plate_says_so(tmp_path, fake_tracker):
    video = scene_video(tmp_path)
    result = read_plates_video(video, tmp_path / "out", model=ScriptedDetector(car), plate_models=(FakePlateDetector(None), FakeReader()))
    saved = json.loads(result["plates_file"].read_text())
    assert saved["tracks"]["1"]["status"] == NO_PLATE and saved["tracks"]["1"]["plate_crop"] is None and result["output_video"].is_file()


def test_video_function_fails_on_a_missing_file_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(plates, "load_detector", lambda: pytest.fail("the detector was loaded"))
    monkeypatch.setattr(plates, "load_plate_models", lambda settings: pytest.fail("the plate models were loaded"))
    with pytest.raises(FileNotFoundError):
        read_plates_video(tmp_path / "missing.mp4", tmp_path)


def clips_manifest(tmp_path, boxes=True):
    video = scene_video(tmp_path, "evt-a.mp4", n=30)
    manifest = {
        "video_id": "small",
        "clips": [{
            "event_id": "evt-a", "clip_file": video.name, "fps": 30.0, "review_status": "pending",
            "vehicle_boxes": [{"frame_in_clip": k, "box_xyxy": [100, 100, 300, 300]} for k in range(30)] if boxes else [],
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_clips_function_gives_each_event_a_plate_result(tmp_path):
    result = read_plates_from_clips(clips_manifest(tmp_path), tmp_path / "out", plate_models=(FakePlateDetector(), FakeReader()))
    event = result["events"][0]
    assert event["event_id"] == "evt-a" and event["license_plate"]["status"] == READ and event["review_status"] == "pending"
    assert (tmp_path / "out" / "plate_crops" / event["plate_crop"]).is_file()
    assert sum(1 for _ in iter_frames(tmp_path / "out" / "evt-a_output.mp4")) == 30
    assert json.loads(result["plates_file"].read_text())["status_counts"] == {READ: 1}


def test_clips_function_handles_an_unreadable_plate_and_keeps_its_evidence(tmp_path):
    result = read_plates_from_clips(clips_manifest(tmp_path), tmp_path / "out", plate_models=(FakePlateDetector(), FakeReader([""])))
    event = result["events"][0]
    assert event["license_plate"]["status"] == UNREADABLE and event["plate_crop"] is not None


def test_clips_function_with_an_empty_manifest_gives_an_empty_result(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"video_id": "small", "clips": []}))
    result = read_plates_from_clips(path, tmp_path / "out", plate_models=(FakePlateDetector(), FakeReader()))
    assert result["events"] == [] and json.loads(result["plates_file"].read_text())["events"] == []


def test_edit_distance():
    assert edit_distance("QQ7007", "QQ707") == 1 and edit_distance("QA1234", "QB1234") == 1 and edit_distance("", "AB") == 2 and edit_distance("AB", "AB") == 0


LABELS = {
    "a": {"text": "AB1234"}, "b": {"text": "AB1234"}, "c": {"text": "AB1234"}, "d": {"text": "AB1234"}, "e": {"text": "AB1234"}, "f": {"text": "AB1234"},
    "g": {"unreadable": True}, "h": {"no_plate_visible": True}, "i": {"unreadable": True}, "j": {"unsure": True},
}


def test_score_sorts_every_outcome():
    results = {
        "a": {"status": READ, "text": "AB1234"},
        "b": {"status": READ, "text": "AB1284"},
        "c": {"status": UNCERTAIN, "text": "AB1234"},
        "d": {"status": UNCERTAIN, "text": "ZZ"},
        "e": {"status": UNREADABLE, "text": None},
        "g": {"status": READ, "text": "XX9999"},
        "h": {"status": UNCERTAIN, "text": "XX9999"},
        "i": {"status": UNREADABLE, "text": None},
        "j": {"status": READ, "text": "XX9999"},
    }
    score = score_plate_results(results, LABELS)
    assert score["readable_labels"] == 6 and score["correct_read"] == 1 and score["wrong_read"] == 1
    assert score["correct_suggestion"] == 1 and score["wrong_suggestion"] == 1 and score["declined"] == 1 and score["no_plate"] == 1
    assert score["unreadable_labels"] == 3 and score["invented_read"] == 1 and score["invented_suggestion"] == 1 and score["unsure_label"] == 1
    assert score["wrong_text_edit_distances"] == [1, 6]


def test_score_counts_a_track_with_no_result_as_no_plate():
    assert score_plate_results({}, {"a": {"text": "AB1234"}})["no_plate"] == 1


def test_mask_keeps_only_a_prefix_and_the_length():
    assert mask_plate("QQ7007") == "QQ****" and mask_plate(None) is None and mask_plate("A") == "A"


def test_no_real_plate_from_the_private_labels_appears_in_committed_tests_or_code():
    # These files go to GitHub, and the private labels hold real plate numbers.
    root = Path(__file__).resolve().parents[1]
    labels_path = root / "data" / "private_annotations" / "plate_labels.json"
    if not labels_path.exists():
        pytest.skip("Private labels are only on machines that have them.")
    videos = json.loads(labels_path.read_text())["videos"].values()
    real = {t["text"] for video in videos for t in video["tracks"].values() if "text" in t}
    committed = [Path(__file__), root / "src" / "traffic_violation_detection" / "plates.py", root / "README.md"]
    assert real and not [text for text in real for path in committed if text in path.read_text()]


def test_settings_are_json():
    json.dumps(PLATE_SETTINGS)


def test_plate_models_stay_on_the_cpu():
    # On the GPU they change PyTorch's later detections, and with them the tracks.
    pytest.importorskip("onnxruntime")
    pytest.importorskip("fast_plate_ocr")
    pytest.importorskip("open_image_models")
    detector, reader = plates.load_plate_models()
    assert detector.model.get_providers() == ["CPUExecutionProvider"]
    assert reader.model.get_providers() == ["CPUExecutionProvider"]
