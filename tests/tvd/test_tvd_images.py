import cv2
import numpy as np
import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_EVIDENCE, clip_from_entry, notebook_08_clips, notebook_08_records

from tvd.domain import Box
from tvd.drawing import FrameOverlay
from tvd.evidence import Clip, evidence_frames, load_evidence_settings, write_evidence_images

IMAGES = load_evidence_settings(ROOT / "configs" / "tvd" / "evidence.yaml").images
needs_notebook_08 = pytest.mark.skipif(not NOTEBOOK_08_EVIDENCE.is_dir(), reason="Needs the outputs of Notebook 08, as on the GPU server.")


def clip(trigger_in_clip: int, tracked: range, frames: int = 133, path=ROOT / "unused.mp4") -> Clip:
    return Clip(path, 100, 100 + frames, 26.0, 640, 360, 100 + trigger_in_clip, {k: Box(10, 10, 50, 40) for k in tracked})


def test_images_before_at_and_after_the_trigger():
    assert evidence_frames(clip(79, range(22, 133)), 18) == [("before", 61), ("entry", 79), ("after", 97)]


def test_a_vehicle_hidden_at_its_entry_is_shown_where_it_is_first_seen():
    assert evidence_frames(clip(79, range(103, 133)), 18) == [("estimated_entry", 79), ("first_seen", 103), ("after", 121)]


def test_the_images_stay_inside_the_clip():
    assert evidence_frames(clip(5, range(0, 20), frames=20), 18) == [("before", 0), ("entry", 5), ("after", 19)]


class Recorder(FrameOverlay):
    """Draws a white square in the corner, and remembers the frames it was asked for."""

    def __init__(self) -> None:
        self.frames: list[int] = []

    def draw(self, image, frame_index) -> None:
        self.frames.append(frame_index)
        image[:20, :20] = 255


def test_images_are_drawn_with_the_overlay_and_the_box(tmp_path, sample_video):
    overlay = Recorder()
    sample = Clip(sample_video, 0, 240, 24.0, 640, 360, 100, {k: Box(300, 100, 400, 200) for k in range(90, 240)})
    images = write_evidence_images(sample, tmp_path, "evt-1", IMAGES, overlay)
    assert [(i.role, i.frame_in_clip, i.source_frame, i.path.name) for i in images] == [
        ("before", 83, 83, "evt-1_before.jpg"), ("entry", 100, 100, "evt-1_entry.jpg"), ("after", 117, 117, "evt-1_after.jpg"),
    ]
    assert overlay.frames == [83, 100, 117]
    before, entry = cv2.imread(str(images[0].path)), cv2.imread(str(images[1].path))
    assert before[5, 5].min() > 230 and not np.allclose(before[150, 300], [0, 255, 255], atol=60)  # Not tracked within 3 frames of 83: no box.
    assert np.allclose(entry[150, 300], [0, 255, 255], atol=60)  # The box's left edge, in yellow.


def test_two_roles_on_one_frame_give_one_image(tmp_path, sample_video):
    end = Clip(sample_video, 0, 240, 24.0, 640, 360, 239, {})
    assert [i.role for i in write_evidence_images(end, tmp_path, "evt-1", IMAGES)] == ["before", "entry"]


@needs_notebook_08
def test_the_same_images_as_notebook_08_without_an_overlay(tmp_path):
    records = notebook_08_records()
    for entry in notebook_08_clips():
        images = write_evidence_images(clip_from_entry(entry), tmp_path, entry["event_id"], IMAGES)
        expected = records[entry["event_id"]]["evidence"]["frames"]
        assert [(i.role, i.frame_in_clip, i.source_frame) for i in images] == [(f["role"], f["frame_in_clip"], f["source_frame"]) for f in expected]
        assert all(i.path.read_bytes() == (NOTEBOOK_08_EVIDENCE / i.path.name).read_bytes() for i in images)
