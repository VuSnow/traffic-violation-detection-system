import numpy as np
import pytest
from conftest import ROOT
from notebook_data import MODIFIED_VIDEO, NOTEBOOK_08_CLIPS, finding_from_clip, notebook_08_clips

from tvd.config import ConfigError
from tvd.domain import Box
from tvd.evidence import Clip, clip_range, cut_clips, load_evidence_settings
from tvd.rules import Finding
from tvd.tracking.history import TrackPoint
from tvd.video import VideoReader

EVIDENCE = load_evidence_settings(ROOT / "configs" / "tvd" / "evidence.yaml")
CLIP = EVIDENCE.clip
needs_notebook_08 = pytest.mark.skipif(not (MODIFIED_VIDEO.is_file() and NOTEBOOK_08_CLIPS.is_dir()), reason="Needs traffic_video_modified.mp4 and the outputs of Notebook 08, as on the GPU server.")


def finding(trigger: int, first: int, last: int) -> Finding:
    """A car tracked from ``first`` to ``last``, moving right by 2 px per frame."""
    points = tuple(TrackPoint(k, Box(2 * k, 100, 2 * k + 40, 130), 0.9, "car") for k in range(first, last + 1))
    return Finding("wrong_way", "7", "car", first, trigger, last, "why", {}, points)


def test_the_settings_file_has_the_notebooks_values():
    assert (CLIP.pre_seconds, CLIP.max_pre_seconds, CLIP.post_seconds, CLIP.crf) == (3.0, 10.0, 2.0, 14)
    assert (EVIDENCE.images.offset_seconds, EVIDENCE.images.jpeg_quality, EVIDENCE.records.duplicate_seconds) == (0.7, 90, 5.0)


def test_a_missing_setting_is_named(tmp_path):
    path = tmp_path / "evidence.yaml"
    path.write_text((ROOT / "configs" / "tvd" / "evidence.yaml").read_text().replace("  crf: 14", ""))
    with pytest.raises(ConfigError, match="clip.crf: Field required"):
        load_evidence_settings(path)


def test_a_clip_starts_3_s_before_the_trigger_and_ends_2_s_after_it():
    assert clip_range(100, 90, 10.0, 0, 1000, CLIP) == (70, 121)


def test_a_clip_follows_a_track_back_to_its_start_but_no_more_than_10_s():
    assert clip_range(100, 40, 10.0, 0, 1000, CLIP) == (40, 121)
    assert clip_range(500, 0, 10.0, 0, 1000, CLIP) == (400, 521)


def test_a_clip_stays_inside_the_analysed_frames():
    assert clip_range(5, 0, 10.0, 0, 15, CLIP) == (0, 15)


def test_a_trigger_outside_the_analysed_frames_is_refused():
    with pytest.raises(ValueError, match="outside"):
        clip_range(20, 0, 10.0, 0, 15, CLIP)


def test_the_same_ranges_as_the_notebooks():
    from traffic_violation_detection.evidence import CLIP_SETTINGS, clip_frame_range

    cases = [(t, s, fps, lo, hi) for t in (5, 50, 200, 700) for s in (0, 40, 190, 650) for fps in (10.0, 26.333, 59.94) for lo, hi in ((0, 715), (30, 705))]
    assert [clip_range(*c, CLIP) for c in cases if c[3] <= c[0] < c[4]] == [clip_frame_range(*c, CLIP_SETTINGS) for c in cases if c[3] <= c[0] < c[4]]


def test_clips_are_cut_in_one_pass(tmp_path, sample_video):
    with VideoReader(sample_video) as reader:
        clips = cut_clips(reader, {"a": finding(100, 90, 130), "b": finding(110, 60, 200)}, 0, 240, tmp_path, CLIP)
        source = {f.index: f.image for f in reader.frames(28, 159)}
    a, b = clips["a"], clips["b"]
    assert (a.start, a.end, a.frame_count, a.trigger_in_clip) == (28, 149, 121, 72) and (b.start, b.end) == (38, 159)
    assert sorted(a.boxes) == list(range(62, 103)) and a.boxes[62] == Box(180, 100, 220, 130)  # Tracked from frame 90.
    with VideoReader(a.path) as clip:
        frames = list(clip.frames())
    assert len(frames) == 121 and np.abs(frames[0].image.astype(int) - source[28]).mean() < 2


def test_the_box_nearest_a_frame_of_the_clip():
    boxes = {k: Box(k, 0, k + 10, 10) for k in range(41) if k != 5}
    clip = Clip(ROOT / "unused.mp4", 0, 41, 10.0, 640, 360, 20, boxes)
    assert clip.box_near(5, 3) == boxes[4] and clip.box_near(43, 3) == boxes[40] and clip.box_near(45, 3) is None


def test_no_clip_is_left_when_the_video_ends_too_soon(tmp_path, sample_video):
    with VideoReader(sample_video) as reader, pytest.raises(RuntimeError, match="ended before"):
        cut_clips(reader, {"a": finding(100, 90, 130), "late": finding(235, 200, 239)}, 0, 300, tmp_path, CLIP)
    assert list(tmp_path.iterdir()) == []


@needs_notebook_08
def test_the_same_clips_as_notebook_08(tmp_path):
    entries = notebook_08_clips()
    with VideoReader(MODIFIED_VIDEO) as reader:
        clips = cut_clips(reader, {e["event_id"]: finding_from_clip(e) for e in entries}, 0, 715, tmp_path, CLIP)
    for entry in entries:
        clip = clips[entry["event_id"]]
        assert (clip.start, clip.end, clip.trigger_in_clip) == (entry["source_start_frame"], entry["source_end_frame"], entry["trigger_frame_in_clip"])
        assert {k: b.as_list() for k, b in clip.boxes.items()} == {b["frame_in_clip"]: b["box_xyxy"] for b in entry["vehicle_boxes"]}
        assert clip.path.read_bytes() == (NOTEBOOK_08_CLIPS / entry["clip_file"]).read_bytes()
