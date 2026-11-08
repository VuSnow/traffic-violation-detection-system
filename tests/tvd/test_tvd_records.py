from dataclasses import replace
from pathlib import Path

import pytest
from conftest import ROOT
from notebook_data import (
    NOTEBOOK_07_REVERSED, NOTEBOOK_08, REVERSED_CANDIDATES, clip_from_entry, finding_from_clip, finding_from_event, load_json,
    notebook_08_clips, notebook_08_plates, notebook_08_records, plate_read_from_json,
)

from tvd.camera import load_rule_defaults
from tvd.evidence import Clip, EvidenceImage
from tvd.plates import READ, UNCERTAIN, PlateResult, load_plates_config, vote
from tvd.records import RecordContext, build_record, deduplicate_records, parse_recorded_at, time_at, validate_record
from tvd.rules import Finding

RULES = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
READING = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml").reading
SETTINGS = {"perception": {"tracker": "ocsort"}, "rules": RULES, "plates": {"version": 1}, "evidence": {"version": 1}}
MODELS = {"detector": {"type": "yolo", "weights": "yolo26m.pt"}, "tracker": {"type": "ocsort"}, "plates": {"type": "onnx_worker"}, "software": {"numpy": "2.4.6"}, "code": None}
RECORDED_AT = parse_recorded_at("2026-11-07T08:00:00+07:00")
FOLDER = Path("/jobs/42")
needs_notebook_outputs = pytest.mark.skipif(
    not (NOTEBOOK_08.is_dir() and NOTEBOOK_07_REVERSED.is_file() and REVERSED_CANDIDATES.is_file()), reason="Needs the outputs of Notebooks 05, 07 and 08, as on the GPU server.",
)


def context(video_id: str = "cam1-0800", fps: float = 25.0) -> RecordContext:
    return RecordContext(video_id, fps, 1920, 1080, "cam1", RECORDED_AT, f"{video_id}/ocsort", SETTINGS, {"camera": "sha256:0123456789ab"}, MODELS)


def red_finding(track_id: str = "7", trigger: int = 250, **evidence) -> Finding:
    measured = {"lane": "through", "signal_state": "red", "entry_seen": True, "entry_estimated": False, "signal_source": "camera"}
    return Finding("red_light", track_id, "car", 200, trigger, 300, "Entered on red.", {**measured, **evidence})


def plate(status: str = READ, text: str | None = "TEST1234") -> PlateResult:
    return PlateResult(status, text, 3, 3, 3, 1.0, 0.99, {}, ())


def record(finding: Finding | None = None, plate_result: PlateResult | None = None, video_id: str = "cam1-0800", fps: float = 25.0) -> dict:
    finding = finding or red_finding()
    clip = Clip(FOLDER / "clips" / "evt.mp4", 175, 301, fps, 1920, 1080, finding.trigger_frame, {})
    images = [EvidenceImage("entry", 75, 250, FOLDER / "evidence" / "evt_entry.jpg")]
    plate_image = FOLDER / "plates" / "evt_f00075.png" if plate_result else None
    return build_record(context(video_id, fps), finding, clip, images, plate_result, plate_image, FOLDER)


def shared_with_version_1(r: dict) -> dict:
    """The fields that version 2 keeps from version 1, unchanged."""
    return {
        "violation_type": r["violation_type"], "track": {k: r["track"][k] for k in ("track_id", "class_name")},
        "trigger": (r["time"]["trigger_frame"], r["time"]["trigger_time_s"]), "rule": {k: r["rule"][k] for k in ("name", "version")},
        "why": r["why"], "signal_source": r["signal_source"], "measurements": r["measurements"], "license_plate": r["license_plate"],
        "needs_attention": r["needs_attention"], "review": r["review"],
    }


def test_recorded_at_needs_a_time_zone():
    assert parse_recorded_at("2026-11-07T01:00:00Z").utcoffset().total_seconds() == 0
    with pytest.raises(ValueError, match="needs a time zone"):
        parse_recorded_at("2026-11-07T08:00:00")
    with pytest.raises(ValueError, match="not an ISO 8601 time"):
        parse_recorded_at("yesterday morning")


def test_absolute_times_keep_the_time_zone_sent():
    assert time_at(RECORDED_AT, 12.038) == "2026-11-07T08:00:12.038+07:00"
    assert time_at(parse_recorded_at("2026-11-07T23:59:59+00:00"), 2.5) == "2026-11-08T00:00:01.500+00:00"
    assert time_at(RECORDED_AT, 70 / 24) == "2026-11-07T08:00:02.917+07:00"  # Rounded, not cut: 2.9167 s.


def test_a_record_is_pending_with_its_times_evidence_and_settings():
    r = record(plate_result=plate())
    assert r["event_id"] == "evt-cam1-0800-red_light-f00250-t7" and r["camera_id"] == "cam1" and r["video"]["recorded_at"] == "2026-11-07T08:00:00+07:00"
    assert r["time"] == {
        "start_frame": 200, "trigger_frame": 250, "end_frame": 300, "start_time_s": 8.0, "trigger_time_s": 10.0, "end_time_s": 12.0,
        "start_at": "2026-11-07T08:00:08.000+07:00", "trigger_at": "2026-11-07T08:00:10.000+07:00", "end_at": "2026-11-07T08:00:12.000+07:00",
    }
    assert r["evidence"] == {
        "clip": "clips/evt.mp4", "clip_frames": [175, 301], "plate_crop": "plates/evt_f00075.png",
        "frames": [{"role": "entry", "frame_in_clip": 75, "source_frame": 250, "image": "evidence/evt_entry.jpg"}],
    }
    assert r["rule"] == {"name": "red_light", "version": 2} and r["settings"]["rule"] == RULES["red_light"] and r["config_files"] == {"camera": "sha256:0123456789ab"}
    assert r["models"] == MODELS
    assert r["signal_source"] == "camera" and "signal_source" not in r["measurements"] and r["needs_attention"] == []
    assert r["review"] == {"status": "pending", "reviewer": None, "decided_at": None, "corrected_plate": None, "notes": None}
    assert validate_record(r) == []


def test_without_a_plate_result_the_record_says_plate_reading_was_not_attempted():
    r = record(plate_result=None)
    assert r["license_plate"]["status"] == "not_attempted" and r["license_plate"]["text"] is None and r["evidence"]["plate_crop"] is None
    assert r["needs_attention"] == ["plate_not_attempted"] and validate_record(r) == []


def test_what_the_reviewer_should_check_is_flagged():
    hidden = red_finding(signal_source="simulated_overlay", entry_estimated=True, entry_seen=False)
    assert record(hidden, plate(UNCERTAIN))["needs_attention"] == ["simulated_signal", "entry_estimated", "plate_uncertain"]
    assert record(red_finding(entry_seen=False), plate())["needs_attention"] == ["entry_not_seen"]


@pytest.mark.parametrize("change, problem", [
    (lambda r: r["review"].update(status="approved", reviewer="someone"), "pending, with no reviewer"),
    (lambda r: r["evidence"].update(clip="/jobs/42/clips/evt.mp4"), "relative"),
    (lambda r: r["evidence"]["frames"][0].update(image="../other/evt_entry.jpg"), "relative"),
    (lambda r: r["time"].update(trigger_at="2026-11-07T08:00:10"), "time zone"),
    (lambda r: r["license_plate"].update(text="TEST1234"), "does not fit its status"),
    (lambda r: r.pop("camera_id"), "camera_id is missing"),
    (lambda r: r.pop("models"), "models is missing"),
])
def test_a_record_unfit_for_review_is_refused(change, problem):
    r = record()
    change(r)
    assert any(problem in p for p in validate_record(r))


def test_one_vehicle_shortly_after_itself_is_one_occurrence():
    first, again, later = record(red_finding("7", 250), plate()), record(red_finding("7", 325), plate()), record(red_finding("7", 400), plate())
    kept, dropped = deduplicate_records([again, first, later], 5.0)
    assert [r["event_id"] for r in kept] == [first["event_id"], later["event_id"]] and dropped[0]["duplicate_of"] == first["event_id"]


def test_two_tracks_with_one_read_plate_are_one_vehicle_but_an_uncertain_plate_is_not_enough():
    assert len(deduplicate_records([record(red_finding("7", 250), plate()), record(red_finding("9", 260), plate())], 5.0)[0]) == 1
    uncertain = [record(red_finding("7", 250), plate(UNCERTAIN)), record(red_finding("9", 260), plate(UNCERTAIN))]
    assert len(deduplicate_records(uncertain, 5.0)[0]) == 2


@needs_notebook_outputs
def test_the_same_red_light_records_as_notebook_08(tmp_path):
    events = {e["event_id"]: e for e in load_json(NOTEBOOK_08 / "violations" / "traffic_video_modified_events.json")["events"]}
    plates, expected = notebook_08_plates(), notebook_08_records()
    for entry in notebook_08_clips():
        name = entry["event_id"]
        finding = finding_from_event(events[name], finding_from_clip(entry).points)
        plate_result = vote([plate_read_from_json(r) for r in plates[name]["license_plate"]["reads"]], READING)
        clip = replace(clip_from_entry(entry), path=tmp_path / entry["clip_file"])  # Inside the folder of the records.
        ours = build_record(context("traffic_video_modified", entry["fps"]), finding, clip, [], plate_result, None, tmp_path)
        matches = shared_with_version_1(ours) == shared_with_version_1(expected[name])  # Outside the assert: a failure never prints a real plate.
        assert matches and validate_record(ours) == [], name


@needs_notebook_outputs
def test_the_same_wrong_way_records_as_notebook_07(tmp_path):
    expected = {r["event_id"]: r for r in load_json(NOTEBOOK_07_REVERSED)["events"]}
    fps = next(iter(expected.values()))["video"]["fps"]
    for event in load_json(REVERSED_CANDIDATES)["candidates"]:
        finding = finding_from_event(event)
        clip = Clip(tmp_path / "clip.mp4", 0, 1, fps, 1920, 1080, finding.trigger_frame, {})
        ours, theirs = build_record(context("video-test-reversed", fps), finding, clip, [], None, None, tmp_path), expected[event["event_id"]]
        assert shared_with_version_1(ours) == shared_with_version_1(theirs) and validate_record(ours) == []
        assert {k: ours["time"][k] for k in theirs["time"]} == theirs["time"]  # Wrong way has a start and an end in both versions.
