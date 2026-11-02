"""Tests for traffic_violation_detection.pipeline, on small drawn videos with stand-ins for every model."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fakes import PassThroughTracker, ScriptedDetector, write_video

from traffic_violation_detection import pipeline, tracking
from traffic_violation_detection.events import APPROVED, PENDING, REJECTED, SqliteViolationStore, load_records
from traffic_violation_detection.pipeline import (
    StepTimes,
    apply_review,
    captions_at,
    check_identities,
    draw_caption,
    run_pipeline,
    scene_kind,
    source_frames,
    store_decisions,
    write_final_video,
)
from traffic_violation_detection.plates import NO_PLATE, READ, UNREADABLE
from traffic_violation_detection.video import iter_frames

WIDTH, HEIGHT = 320, 240
RED_LIGHT_SCENE = Path(__file__).resolve().parents[1] / "configs" / "scenes" / "rlr_occasions.json"
WRONG_WAY_SCENE = Path(__file__).resolve().parents[1] / "configs" / "scenes" / "video-test.json"


class Plates:
    """Stand-ins for the plate detector and the OCR, which read one plate on any vehicle."""

    def __init__(self, text="AB1234", found=True, fails=False):
        self.text, self.found, self.fails = text, found, fails

    def predict(self, crop):
        if self.fails:
            raise RuntimeError("worker stopped")
        height, width = crop.shape[:2]
        box = SimpleNamespace(x1=int(0.2 * width), y1=int(0.7 * height), x2=int(0.8 * width), y2=int(0.9 * height))
        return [SimpleNamespace(confidence=0.9, bounding_box=box)] if self.found else []

    def run_one(self, image, return_confidence=False, remove_pad_char=True):
        return SimpleNamespace(plate=self.text.ljust(10, "_"), char_probs=np.full(10, 0.99))


def models(**changes):
    fake = Plates(**changes)
    return fake, fake


def frame(color):
    image = np.full((HEIGHT, WIDTH, 3), 40, np.uint8)
    cv2.rectangle(image, (20, 20), (60, 60), color, -1)
    return image


RED, GREEN = (30, 30, 230), (230, 220, 40)


def scene(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene_id": "small", "video": {"width": WIDTH, "height": HEIGHT}, "lane_direction_deg": -90.0,
        "clips": {"main": {
            "start": 0, "end": 60, "stop_line": [[0, 120], [WIDTH, 120]], "lane_dividers": [10, 310],
            "signals": {"method": "head", "through": [[20, 20, 60, 60]], "left": [[20, 20, 60, 60]]},
        }},
    }))
    return path


def car(call):
    # A car driving up the image, large enough for its plate to be read, that crosses the line near call 29.
    bottom = 220 - 3 * call
    return [(100, bottom - 60, 220, bottom, 0.9, 2)] if bottom > 65 else []


@pytest.fixture(autouse=True)
def fake_tracker(monkeypatch):
    monkeypatch.setattr(tracking, "create_tracker", PassThroughTracker)


def run(tmp_path, color=RED, script=car, plate_models=None, **arguments):
    video = write_video(tmp_path / "small.mp4", [frame(color)] * 60)
    return run_pipeline(video, tmp_path / "out", scene(tmp_path), model=ScriptedDetector(script), plate_models=plate_models or models(), **arguments)


def test_a_video_goes_from_start_to_finish(tmp_path):
    result = run(tmp_path)
    folder = tmp_path / "out"
    assert result["output_video"] == folder / "small_output.mp4" and sum(1 for _ in iter_frames(result["output_video"])) == 60
    assert len(result["records"]) == 1 and result["records"][0]["license_plate"]["text"] == "AB1234"
    for path in ("violations/small_events.json", "plates/plates.json", "events/small_review_events.json", "small_summary.json"):
        assert (folder / path).is_file()


def test_without_a_review_everything_stays_pending_and_nothing_is_stored(tmp_path):
    result = run(tmp_path)
    assert result["pending"] == 1 and result["database"] is None and result["summary"]["stored"] == 0
    assert not (tmp_path / "out" / "violations.db").exists()
    assert load_records(result["records_file"])[0]["review"]["status"] == PENDING


def test_summary_tells_what_happened_and_how_long_it_took(tmp_path):
    summary = run(tmp_path)["summary"]
    assert summary["rule"] == "red_light" and summary["frames_analysed"] == 60 and summary["candidates"] == 1
    assert summary["plates"] == {READ: 1} and summary["review"] == {PENDING: 1} and summary["failures"] == []
    assert set(summary["steps_seconds"]) >= {"detect, track and apply the rule", "read plates", "build records and evidence", "write the final video"}
    assert summary["total_seconds"] >= 0 and json.loads((tmp_path / "out" / "small_summary.json").read_text()) == summary


def test_a_video_with_no_violation_finishes_cleanly(tmp_path):
    result = run(tmp_path, color=GREEN)
    assert result["records"] == [] and result["summary"]["candidates"] == 0 and result["summary"]["outcomes"] == {"signal_not_red": 1}
    assert result["output_video"].is_file()


def test_a_video_with_no_vehicle_finishes_cleanly(tmp_path):
    result = run(tmp_path, script=lambda call: [])
    assert result["records"] == [] and result["summary"]["outcomes"] == {} and sum(1 for _ in iter_frames(result["output_video"])) == 60


def test_an_unreadable_plate_is_stated_and_the_violation_stays(tmp_path):
    result = run(tmp_path, plate_models=models(text=""))
    assert result["records"][0]["license_plate"]["status"] == UNREADABLE and "plate_unreadable" in result["records"][0]["needs_attention"]


def test_no_plate_is_stated_and_the_violation_stays(tmp_path):
    result = run(tmp_path, plate_models=models(found=False))
    assert result["records"][0]["license_plate"]["status"] == NO_PLATE and len(result["records"]) == 1


def test_a_failed_plate_step_keeps_the_violations(tmp_path):
    result = run(tmp_path, plate_models=models(fails=True))
    record = result["records"][0]
    assert record["license_plate"]["status"] == "not_attempted" and "plate_not_attempted" in record["needs_attention"]
    assert "worker stopped" in result["summary"]["failures"][0]


def test_plate_models_that_cannot_load_keep_the_violations(tmp_path, monkeypatch):
    def cannot_load():
        raise RuntimeError("Plate reading needs its models")

    monkeypatch.setattr(pipeline, "load_plate_models", cannot_load)
    video = write_video(tmp_path / "small.mp4", [frame(RED)] * 60)
    result = run_pipeline(video, tmp_path / "out", scene(tmp_path), model=ScriptedDetector(car))
    assert len(result["records"]) == 1 and "needs its models" in result["summary"]["failures"][0]


def test_a_missing_video_fails_before_any_model_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda: pytest.fail("the detector was loaded"))
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "missing.mp4", tmp_path / "out", scene(tmp_path))


def test_a_scene_of_another_size_fails_before_any_model_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda: pytest.fail("the detector was loaded"))
    video = write_video(tmp_path / "other.mp4", [np.zeros((100, 200, 3), np.uint8)] * 3)
    with pytest.raises(ValueError, match="video is 200x100"):
        run_pipeline(video, tmp_path / "out", scene(tmp_path))


def approve(record):
    return {"decision": APPROVED, "reviewer": "reviewer-1"}


def reject(record):
    return {"decision": REJECTED, "reviewer": "reviewer-1", "notes": "Stopped."}


def test_an_approved_event_reaches_the_database(tmp_path):
    result = run(tmp_path, review=approve)
    assert result["summary"]["stored"] == 1 and result["summary"]["review"] == {APPROVED: 1}
    saved = SqliteViolationStore(result["database"]).find_by_plate("AB1234")
    assert saved[0]["reviewer"] == "reviewer-1" and saved[0]["event_id"] == result["records"][0]["event_id"]


def test_a_rejected_event_leaves_only_an_audit_record(tmp_path):
    result = run(tmp_path, review=reject)
    assert result["summary"]["stored"] == 0 and len(result["audit"]) == 1
    audit = (tmp_path / "out" / "small_audit.json").read_text()
    assert "Stopped." in audit and "AB1234" not in audit


def test_a_review_may_leave_a_record_pending(tmp_path):
    result = run(tmp_path, review=lambda record: None)
    assert result["pending"] == 1 and result["summary"]["stored"] == 0


def test_the_records_on_file_stay_pending_after_a_review(tmp_path):
    result = run(tmp_path, review=approve)
    assert load_records(result["records_file"])[0]["review"]["status"] == PENDING


def test_running_again_replaces_the_database(tmp_path):
    run(tmp_path, review=approve)
    result = run(tmp_path, review=approve)
    assert SqliteViolationStore(result["database"]).count() == 1


def test_running_again_gives_the_same_event_ids(tmp_path):
    first = run(tmp_path)["records"][0]["event_id"]
    assert run(tmp_path)["records"][0]["event_id"] == first


def test_review_receives_each_record_once():
    seen = []
    records = [{"event_id": "a", "review": {"status": PENDING}}]
    apply_review(records, lambda record: seen.append(record["event_id"]), "2026-11-02T10:00:00+00:00")
    assert seen == ["a"]


def test_store_decisions_replaces_an_old_file(tmp_path):
    path = tmp_path / "violations.db"
    path.write_bytes(b"")
    assert store_decisions([], path) == (0, [])


def test_identities_that_stay_the_same_pass():
    ids = {"a", "b"}
    assert check_identities(ids, ids, ids, {"a"}, {"b"}) == []


def test_identities_without_a_plate_step_pass():
    assert check_identities({"a"}, {"a"}, None, {"a"}, set()) == []


@pytest.mark.parametrize(
    "clips, plates, records, message",
    [({"a"}, {"a", "b"}, {"a", "b"}, "clips differ"), ({"a", "b"}, {"a"}, {"a", "b"}, "plates differ"), ({"a", "b"}, {"a", "b"}, {"a"}, "records differ")],
)
def test_an_event_lost_between_steps_is_caught(clips, plates, records, message):
    assert any(message in problem for problem in check_identities({"a", "b"}, clips, plates, records, set()))


def test_rule_comes_from_the_scene():
    assert scene_kind(RED_LIGHT_SCENE) == "red_light" and scene_kind(WRONG_WAY_SCENE) == "wrong_way"


def test_source_frames_follow_the_clips_of_a_red_light_scene(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"clips": {"a": {"start": 2, "end": 4}, "b": {"start": 10, "end": 12}}}))
    assert source_frames(path, "red_light", 0) == [2, 3, 10, 11]


def test_source_frames_of_a_wrong_way_scene_are_the_whole_video():
    assert source_frames(WRONG_WAY_SCENE, "wrong_way", 3) == [0, 1, 2]


def record_at(frame_number, status=READ, text="AB1234"):
    return {
        "violation_type": "red_light", "time": {"trigger_frame": frame_number, "trigger_time_s": frame_number / 30.0},
        "license_plate": {"status": status, "text": text}, "review": {"status": PENDING},
    }


def test_caption_appears_for_a_few_seconds_after_the_event():
    records = [record_at(30)]
    assert captions_at(29, records, 30.0) == [] and captions_at(119, records, 30.0) != [] and captions_at(120, records, 30.0) == []
    assert "AB1234" in captions_at(30, records, 30.0)[0] and "pending" in captions_at(30, records, 30.0)[0]


def test_caption_says_when_a_plate_is_only_a_suggestion():
    assert "AB1234?" in captions_at(30, [record_at(30, "uncertain")], 30.0)[0]


def test_caption_marks_the_bottom_of_the_frame():
    image = np.full((240, 320, 3), 200, np.uint8)
    draw_caption(image, ["one line", "another"])
    assert image[235, 300].tolist() == [200, 200, 200] and image[235, 2].tolist() == [0, 0, 0] and image[20, 2].tolist() == [200, 200, 200]


def test_final_video_shows_the_caption_on_the_right_frames(tmp_path):
    annotated = write_video(tmp_path / "annotated.mp4", [np.full((HEIGHT, WIDTH, 3), 200, np.uint8)] * 20)
    output = tmp_path / "final.mp4"
    # The video holds source frames 100 to 119, and the event happens at source frame 110.
    write_final_video(annotated, output, list(range(100, 120)), [record_at(110)], 30.0)
    frames = [r["image"] for r in iter_frames(output)]
    assert frames[5][235, 2].mean() > 150 and frames[12][235, 2].mean() < 50


def test_final_video_with_no_event_is_the_annotated_video(tmp_path):
    annotated = write_video(tmp_path / "annotated.mp4", [np.zeros((HEIGHT, WIDTH, 3), np.uint8)] * 3)
    write_final_video(annotated, tmp_path / "final.mp4", [0, 1, 2], [], 30.0)
    assert (tmp_path / "final.mp4").read_bytes() == annotated.read_bytes()


def test_step_times_add_up_and_survive_an_error():
    times = StepTimes()
    with times("a"):
        pass
    with pytest.raises(ValueError):
        with times("b"):
            raise ValueError
    assert set(times.seconds) == {"a", "b"}
