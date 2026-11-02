"""Tests for traffic_violation_detection.events, on small clips and made-up records."""

import copy
import json

import cv2
import numpy as np
import pytest
from fakes import write_video

from traffic_violation_detection.events import (
    APPROVED,
    NOT_ATTEMPTED,
    PENDING,
    REJECTED,
    SqliteViolationStore,
    attention_flags,
    box_near,
    build_record,
    confirmed_record,
    decide,
    deduplicate_records,
    evidence_frame_numbers,
    final_plate,
    generate_review_events,
    load_records,
    plate_caption,
    plate_summary,
    rejection_record,
    same_occurrence,
    save_records,
    validate_record,
    write_evidence_frames,
)
from traffic_violation_detection.plates import NO_PLATE, READ, UNCERTAIN, UNREADABLE
from traffic_violation_detection.video import iter_frames

VIDEO = {"video_id": "small", "path": "small.mp4", "fps": 30.0, "width": 320, "height": 240, "reported_frame_count": 90, "reported_duration_s": 3.0}
NOW = "2026-11-02T10:00:00+00:00"


def event(frame=30, track="4", event_id=None, **changes):
    return {
        "event_id": event_id or f"evt-small-main-red_light-f{frame:05d}-t{track}", "video_id": "small", "clip": "main", "violation_type": "red_light",
        "track_id": track, "class_name": "car", "trigger_frame": frame, "trigger_time_s": frame / 30.0, "time_in_clip_s": frame / 30.0,
        "rule": {"name": "red_light", "version": 2}, "signal_source": "camera", "why": "Entered on red.",
        "evidence": {"x": 100.0, "lane": "through", "signal_group": "through", "signal_state": "red", "movement": "straight", "entry_seen": True, "entry_estimated": False, "box_xyxy": [1, 2, 3, 4]},
        "license_plate": None, "review_status": PENDING, **changes,
    }


def plate(status=READ, text="AB1234"):
    return {"status": status, "text": text, "votes": 4, "usable_reads": 4, "agreement": 1.0, "mean_prob": 0.99, "plates_found": 5, "other_texts": {}, "reads": []}


def record(frame=30, track="4", plate_result=None, **changes):
    return build_record(event(frame, track, **changes), VIDEO, "small/ocsort-byte", plate_result or plate())


def test_record_joins_the_rule_the_track_the_time_and_the_plate():
    built = record()
    assert built["violation_type"] == "red_light" and built["track"]["track_id"] == "4" and built["track"]["tracking_session"] == "small/ocsort-byte"
    assert built["time"]["trigger_frame"] == 30 and built["time"]["trigger_time_s"] == 1.0 and built["time"]["start_frame"] is None
    assert built["license_plate"]["text"] == "AB1234" and built["rule"]["version"] == 2 and built["why"]
    json.dumps(built)


def test_every_record_starts_pending_with_no_reviewer():
    assert record()["review"] == {"status": PENDING, "reviewer": None, "decided_at": None, "corrected_plate": None, "notes": None}


def test_wrong_way_event_fits_the_same_record():
    wrong_way = event(track="7", violation_type="wrong_way", start_frame=10, start_time_s=0.3, end_frame=50, end_time_s=1.6, region_id="right", tracking_session="small/ocsort-byte")
    del wrong_way["clip"]
    built = build_record(wrong_way, VIDEO, "ignored", None)
    assert validate_record(built) == [] and built["time"]["end_frame"] == 50 and built["measurements"]["region_id"] == "right"
    assert built["license_plate"]["status"] == NOT_ATTEMPTED and built["track"]["tracking_session"] == "small/ocsort-byte"


def test_no_plate_result_is_stated_not_left_out():
    assert plate_summary(None)["status"] == NOT_ATTEMPTED


def test_plate_summary_drops_the_reads():
    assert "reads" not in plate_summary(plate())


@pytest.mark.parametrize(
    "plate_result, changes, flag",
    [
        (plate(UNCERTAIN), {}, "plate_uncertain"),
        (plate(UNREADABLE, None), {}, "plate_unreadable"),
        (plate(NO_PLATE, None), {}, "plate_not_found"),
        (plate(), {"signal_source": "simulated_overlay"}, "simulated_signal"),
    ],
)
def test_reviewer_is_told_what_to_check(plate_result, changes, flag):
    assert flag in record(plate_result=plate_result, **changes)["needs_attention"]


def test_estimated_entry_is_flagged_once():
    estimated = event()
    estimated["evidence"].update(entry_seen=False, entry_estimated=True)
    assert attention_flags(estimated, plate_summary(plate())) == ["entry_estimated"]


def test_entry_not_seen_is_flagged():
    fragment = event()
    fragment["evidence"]["entry_seen"] = False
    assert attention_flags(fragment, plate_summary(plate())) == ["entry_not_seen"]


def test_plain_record_needs_no_attention():
    assert record()["needs_attention"] == []


def test_sound_record_has_no_problems():
    assert validate_record(record()) == []


def broken(change):
    built = copy.deepcopy(record())
    change(built)
    return validate_record(built)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r.pop("rule"), "rule is missing"),
        (lambda r: r.update(why=None), "why has the wrong type"),
        (lambda r: r.update(event_id="4"), "must differ from the track id"),
        (lambda r: r.update(event_id="evt-other-1"), "must name its video"),
        (lambda r: r["track"].update(tracking_session="other/ocsort-byte"), "scoped to a tracking session"),
        (lambda r: r["time"].update(trigger_frame=-1), "trigger time"),
        (lambda r: r["license_plate"].update(status="maybe"), "plate status"),
        (lambda r: r["license_plate"].update(text=None), "does not fit its status"),
        (lambda r: r["license_plate"].update(status=UNREADABLE), "does not fit its status"),
        (lambda r: r["review"].update(status="confirmed"), "review status"),
        (lambda r: r["review"].update(reviewer="someone"), "cannot have a reviewer"),
        (lambda r: r["review"].update(status=APPROVED), "needs a reviewer"),
    ],
)
def test_broken_records_are_caught(change, message):
    assert any(message in problem for problem in broken(change))


def test_same_id_is_one_occurrence():
    assert same_occurrence(record(), record())


def test_same_track_shortly_after_is_one_occurrence():
    assert same_occurrence(record(30), record(60))


def test_same_read_plate_on_another_track_is_one_occurrence():
    assert same_occurrence(record(30, "4"), record(60, "9"))


def test_same_plate_much_later_is_another_occurrence():
    assert not same_occurrence(record(30, "4"), record(300, "9"))


def test_a_suggested_plate_does_not_make_two_events_one():
    assert not same_occurrence(record(30, "4", plate(UNCERTAIN)), record(60, "9", plate(UNCERTAIN)))


def test_unreadable_plates_do_not_make_two_events_one():
    assert not same_occurrence(record(30, "4", plate(UNREADABLE, None)), record(60, "9", plate(UNREADABLE, None)))


def test_same_plate_for_another_violation_type_is_another_occurrence():
    assert not same_occurrence(record(30, "4"), record(60, "9", violation_type="wrong_way"))


def test_deduplication_keeps_the_first_and_says_what_was_dropped():
    first, repeat, other = record(30, "4"), record(60, "4"), record(300, "9", plate(UNREADABLE, None))
    kept, dropped = deduplicate_records([repeat, other, first])
    assert [r["event_id"] for r in kept] == [first["event_id"], other["event_id"]]
    assert dropped[0]["event_id"] == repeat["event_id"] and dropped[0]["duplicate_of"] == first["event_id"]


def test_running_a_step_twice_gives_no_extra_records():
    kept, dropped = deduplicate_records([record(), record()])
    assert len(kept) == 1 and len(dropped) == 1


ENTRY = {
    "event_id": "evt-small-main-red_light-f00030-t4", "clip_file": "evt-small-main-red_light-f00030-t4.mp4", "fps": 30.0, "frame_count": 60, "source_start_frame": 100,
    "trigger_frame_in_clip": 30, "vehicle_boxes": [{"frame_in_clip": k, "box_xyxy": [40.0, 40.0, 120.0, 120.0]} for k in range(10, 45)],
    "review_status": PENDING, "license_plate": None, "signal_source": "camera",
}


def small_clip(path, n=60):
    frames = []
    for k in range(n):
        image = np.full((240, 320, 3), 60, np.uint8)
        cv2.circle(image, (20 + 4 * k, 200), 8, (200, 200, 200), -1)
        frames.append(image)
    return write_video(path, frames)


def test_evidence_frames_are_before_at_and_after_the_entry():
    assert evidence_frame_numbers(ENTRY) == [("before", 9), ("entry", 30), ("after", 51)]


def test_evidence_frames_stay_inside_the_clip():
    early = {**ENTRY, "trigger_frame_in_clip": 2, "frame_count": 10, "vehicle_boxes": [{"frame_in_clip": 0, "box_xyxy": [1, 2, 3, 4]}]}
    assert evidence_frame_numbers(early) == [("before", 0), ("entry", 2), ("after", 9)]


def test_vehicle_hidden_at_its_entry_is_shown_when_it_is_first_seen():
    hidden = {**ENTRY, "trigger_frame_in_clip": 5}
    assert evidence_frame_numbers(hidden) == [("estimated_entry", 5), ("first_seen", 10), ("after", 31)]


def test_box_near_finds_a_close_frame_only():
    assert box_near(ENTRY, 9) is not None and box_near(ENTRY, 5) is None and box_near({**ENTRY, "vehicle_boxes": []}, 9) is None


def test_evidence_images_are_saved_with_references(tmp_path):
    clip = small_clip(tmp_path / "clip.mp4")
    frames = write_evidence_frames(clip, ENTRY, tmp_path / "out" / "evidence")
    assert [f["role"] for f in frames] == ["before", "entry", "after"]
    assert frames[1]["source_frame"] == 130 and frames[1]["image"].startswith("evidence/")
    assert all((tmp_path / "out" / f["image"]).is_file() for f in frames)


def test_evidence_image_has_the_vehicle_marked_and_the_clip_stays_clean(tmp_path):
    clip = small_clip(tmp_path / "clip.mp4")
    frames = write_evidence_frames(clip, ENTRY, tmp_path / "out" / "evidence")
    image = cv2.imread(str(tmp_path / "out" / frames[1]["image"]))
    assert image[40, 80, 1] > 150 and image[40, 80, 0] < 110
    clean = list(iter_frames(clip, 30, 31))[0]["image"]
    assert clean[40, 80, 1] < 90


def test_records_survive_saving_and_loading(tmp_path):
    save_records([record()], tmp_path / "events.json")
    assert load_records(tmp_path / "events.json") == [record()]


def test_records_of_another_schema_are_refused(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"schema_version": 99, "events": []}))
    with pytest.raises(ValueError, match="schema_version"):
        load_records(path)


def test_approval_needs_a_reviewer_and_records_the_decision():
    decided = decide(record(), APPROVED, "reviewer-1", NOW, notes="Clear on the clip.")
    assert decided["review"] == {"status": APPROVED, "reviewer": "reviewer-1", "decided_at": NOW, "corrected_plate": None, "notes": "Clear on the clip."}
    assert validate_record(decided) == []


def test_deciding_leaves_the_pending_record_unchanged():
    pending = record()
    decide(pending, APPROVED, "reviewer-1", NOW)
    assert pending["review"]["status"] == PENDING


@pytest.mark.parametrize("reviewer", ["", "   "])
def test_nobody_can_decide_without_a_name(reviewer):
    with pytest.raises(ValueError, match="needs a reviewer"):
        decide(record(), APPROVED, reviewer, NOW)


def test_unknown_decision_is_refused():
    with pytest.raises(ValueError, match="Unknown decision"):
        decide(record(), "confirmed", "reviewer-1", NOW)


def test_decision_needs_a_readable_time():
    with pytest.raises(ValueError):
        decide(record(), APPROVED, "reviewer-1", "yesterday")


@pytest.mark.parametrize("first", [APPROVED, REJECTED])
def test_a_decision_cannot_be_replaced(first):
    decided = decide(record(), first, "reviewer-1", NOW)
    with pytest.raises(ValueError, match="already decided"):
        decide(decided, APPROVED, "reviewer-2", NOW)


def test_sure_plate_is_kept_as_read_by_the_ocr():
    assert final_plate(decide(record(), APPROVED, "r", NOW)) == ("AB1234", "ocr")


def test_reviewers_correction_replaces_the_ocr():
    assert final_plate(decide(record(), APPROVED, "r", NOW, corrected_plate="AB1284")) == ("AB1284", "reviewer")


def test_a_suggestion_is_not_kept_unless_the_reviewer_enters_it():
    pending = record(plate_result=plate(UNCERTAIN))
    assert final_plate(decide(pending, APPROVED, "r", NOW)) == (None, None)
    assert final_plate(decide(pending, APPROVED, "r", NOW, corrected_plate="AB1234")) == ("AB1234", "reviewer")


def test_reviewer_can_supply_a_plate_the_ocr_could_not_read():
    assert final_plate(decide(record(plate_result=plate(UNREADABLE, None)), APPROVED, "r", NOW, corrected_plate="XY999")) == ("XY999", "reviewer")


@pytest.mark.parametrize("status", [PENDING, REJECTED])
def test_only_an_approved_event_can_be_persisted(status):
    candidate = record() if status == PENDING else decide(record(), REJECTED, "r", NOW)
    with pytest.raises(ValueError, match="only an approved event"):
        confirmed_record(candidate)


def test_confirmed_record_says_who_decided_and_how_the_plate_is_known():
    confirmed = confirmed_record(decide(record(), APPROVED, "reviewer-1", NOW, notes="ok"))
    assert confirmed["reviewer"] == "reviewer-1" and confirmed["plate_source"] == "ocr" and confirmed["rule_version"] == 2
    json.dumps(confirmed)


def test_rejection_keeps_the_decision_but_no_plate_or_images():
    kept = rejection_record(decide(record(), REJECTED, "reviewer-1", NOW, notes="Stopped at the line."))
    assert kept["notes"] == "Stopped at the line." and "plate_text" not in kept and "evidence" not in kept and "AB1234" not in json.dumps(kept)


def test_pending_event_has_no_rejection_record():
    with pytest.raises(ValueError, match="not rejected"):
        rejection_record(record())


def confirmed(frame=30, track="4", **changes):
    return confirmed_record(decide(record(frame, track), APPROVED, "reviewer-1", NOW, **changes))


def test_store_gives_back_what_it_saved():
    store = SqliteViolationStore()
    store.save(confirmed())
    saved = store.get(confirmed()["event_id"])
    assert saved == confirmed() and store.count() == 1


def test_store_has_nothing_for_an_unknown_event():
    assert SqliteViolationStore().get("evt-none") is None


def test_store_refuses_the_same_event_twice():
    store = SqliteViolationStore()
    store.save(confirmed())
    with pytest.raises(ValueError, match="already stored"):
        store.save(confirmed())
    assert store.count() == 1


def test_store_refuses_a_candidate_that_went_around_review():
    store = SqliteViolationStore()
    with pytest.raises(ValueError, match="Not a confirmed record"):
        store.save(record())
    with pytest.raises(ValueError, match="Not a confirmed record"):
        store.save({**confirmed(), "reviewer": None})
    assert store.count() == 0


def test_store_finds_violations_by_plate():
    store = SqliteViolationStore()
    store.save(confirmed(30, "4"))
    store.save(confirmed(600, "9"))
    assert len(store.find_by_plate("AB1234")) == 2 and store.find_by_plate("ZZ0000") == []


def test_store_keeps_data_in_a_file(tmp_path):
    SqliteViolationStore(tmp_path / "violations.db").save(confirmed())
    assert SqliteViolationStore(tmp_path / "violations.db").count() == 1


@pytest.mark.parametrize(
    "status, text, caption",
    [(READ, "AB1234", "AB1234"), (UNCERTAIN, "AB1234", "AB1234?"), (UNREADABLE, None, "unreadable"), (NO_PLATE, None, "not found"), (NOT_ATTEMPTED, None, "not attempted")],
)
def test_plate_caption_shows_how_sure_the_result_is(status, text, caption):
    assert plate_caption({"status": status, "text": text}) == caption


def violation_outputs(tmp_path, events, plates=None):
    folder = tmp_path / "violations"
    clips = folder / "small_violation_clips"
    clips.mkdir(parents=True)
    entries = []
    for item in events:
        small_clip(clips / f"{item['event_id']}.mp4")
        entries.append({**ENTRY, "event_id": item["event_id"], "clip_file": f"{item['event_id']}.mp4"})
    (clips / "manifest.json").write_text(json.dumps({"video_id": "small", "clips": entries}))
    (folder / "small_events.json").write_text(json.dumps({"video_id": "small", "tracking_session": "small/ocsort-byte", "signal_source": "camera", "events": events}))
    video = small_clip(tmp_path / "small.mp4", n=90)
    plates_file = None
    if plates is not None:
        plates_file = tmp_path / "plates" / "plates.json"
        (plates_file.parent / "plate_crops").mkdir(parents=True)
        for item in plates:
            if item["plate_crop"]:
                cv2.imwrite(str(plates_file.parent / "plate_crops" / item["plate_crop"]), np.zeros((20, 60, 3), np.uint8))
        plates_file.write_text(json.dumps({"events": plates}))
    return video, folder, plates_file


def plate_event(item, status=READ, text="AB1234", crop="crop.png"):
    return {"event_id": item["event_id"], "license_plate": plate(status, text), "plate_crop": crop}


def test_review_function_saves_records_evidence_and_a_video(tmp_path):
    item = event()
    video, folder, plates_file = violation_outputs(tmp_path, [item], [plate_event(item)])
    result = generate_review_events(video, tmp_path / "out", folder, plates_file)
    assert result["records_file"] == tmp_path / "out" / "small_review_events.json" and result["output_video"] == tmp_path / "out" / "small_output.mp4"
    assert sum(1 for _ in iter_frames(result["output_video"])) == 60
    saved = load_records(result["records_file"])[0]
    assert saved["review"]["status"] == PENDING and saved["license_plate"]["text"] == "AB1234"
    assert (tmp_path / "out" / saved["evidence"]["clip"]).is_file() and (tmp_path / "out" / saved["evidence"]["plate_crop"]).is_file()
    assert all((tmp_path / "out" / f["image"]).is_file() for f in saved["evidence"]["frames"])


def test_review_function_without_plates_says_they_were_not_attempted(tmp_path):
    video, folder, _ = violation_outputs(tmp_path, [event()])
    result = generate_review_events(video, tmp_path / "out", folder)
    assert result["records"][0]["license_plate"]["status"] == NOT_ATTEMPTED and "plate_not_attempted" in result["records"][0]["needs_attention"]


def test_review_function_keeps_an_unreadable_plates_crop_as_evidence(tmp_path):
    item = event()
    video, folder, plates_file = violation_outputs(tmp_path, [item], [plate_event(item, UNREADABLE, None)])
    saved = generate_review_events(video, tmp_path / "out", folder, plates_file)["records"][0]
    assert saved["evidence"]["plate_crop"] is not None and saved["license_plate"]["text"] is None


def test_review_function_with_no_events_saves_an_empty_file_and_no_video(tmp_path):
    video, folder, _ = violation_outputs(tmp_path, [])
    result = generate_review_events(video, tmp_path / "out", folder)
    assert result["records"] == [] and result["output_video"] is None and load_records(result["records_file"]) == []


def test_review_function_reads_the_events_of_the_wrong_way_step(tmp_path):
    item = event(track="7", violation_type="wrong_way")
    video, folder, _ = violation_outputs(tmp_path, [item])
    saved = json.loads((folder / "small_events.json").read_text())
    saved["candidates"] = saved.pop("events")
    (folder / "small_events.json").write_text(json.dumps(saved))
    assert generate_review_events(video, tmp_path / "out", folder)["records"][0]["violation_type"] == "wrong_way"


def test_review_function_drops_a_duplicate(tmp_path):
    first, repeat = event(30, "4"), event(60, "4")
    video, folder, _ = violation_outputs(tmp_path, [first, repeat])
    result = generate_review_events(video, tmp_path / "out", folder)
    assert len(result["records"]) == 1 and result["duplicates"][0]["duplicate_of"] == first["event_id"]


def test_review_function_removes_the_images_of_an_earlier_run(tmp_path):
    video, folder, _ = violation_outputs(tmp_path, [event()])
    stale = tmp_path / "out" / "evidence" / "stale_entry.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    generate_review_events(video, tmp_path / "out", folder)
    assert not stale.exists()


def test_review_function_stops_on_a_broken_record(tmp_path):
    item = event(event_id="evt-other-1")
    video, folder, _ = violation_outputs(tmp_path, [item])
    with pytest.raises(ValueError, match="not fit for review"):
        generate_review_events(video, tmp_path / "out", folder)
    assert not (tmp_path / "out" / "small_review_events.json").exists()
