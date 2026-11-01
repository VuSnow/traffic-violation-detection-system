"""Tests for traffic_violation_detection.evidence: frame ranges, and real clips cut from a committed video."""

import builtins
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_violation_detection import evidence
from traffic_violation_detection.evidence import clip_frame_range, cut_event_clip, find_ffmpeg, write_clip
from traffic_violation_detection.video import read_frame_at

VIDEO = Path(__file__).resolve().parents[1] / "data" / "traffic_sample.mp4"  # 640x360, 24 FPS, 240 frames
FPS = 30.0


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255**2 / mse)


def decoded(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_range_reaches_back_to_the_start_of_the_track():
    assert clip_frame_range(1000, 800, FPS, 0, 5000) == (800, 1061)


def test_range_has_a_minimum_lead_for_a_short_track():
    assert clip_frame_range(1000, 990, FPS, 0, 5000)[0] == 910


def test_range_follows_a_long_track_back_only_so_far():
    assert clip_frame_range(1000, 100, FPS, 0, 5000)[0] == 700


def test_range_stays_inside_the_limits():
    assert clip_frame_range(1000, 800, FPS, 950, 1020) == (950, 1020)


@pytest.mark.parametrize("trigger", [949, 1020])
def test_range_rejects_a_trigger_outside_the_limits(trigger):
    with pytest.raises(ValueError, match="outside"):
        clip_frame_range(trigger, 900, FPS, 950, 1020)


def test_ffmpeg_missing_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    real_import = builtins.__import__

    def no_imageio(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_imageio)
    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        find_ffmpeg()


def test_ffmpeg_falls_back_to_the_package(monkeypatch):
    package = pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert find_ffmpeg() == package.get_ffmpeg_exe()


def test_clip_has_exactly_the_requested_frames_in_order(tmp_path):
    out = tmp_path / "clip.mp4"
    assert write_clip(VIDEO, 50, 80, out) == 30
    frames = decoded(out)
    assert len(frames) == 30
    for i in (0, 15, 29):
        source = read_frame_at(VIDEO, 50 + i)["image"]
        assert psnr(frames[i], source) >= 35
        # This video burns its frame number into the image, so another frame must match worse.
        assert psnr(frames[i], source) > psnr(frames[i], read_frame_at(VIDEO, 50 + i + 12)["image"])


def test_clip_keeps_the_fps_and_size_of_the_source(tmp_path):
    out = tmp_path / "clip.mp4"
    write_clip(VIDEO, 0, 24, out)
    capture = cv2.VideoCapture(str(out))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(24.0)
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (640, 360)
    capture.release()


def test_clip_is_not_blurred(tmp_path):
    out = tmp_path / "clip.mp4"
    write_clip(VIDEO, 100, 110, out)
    sharpness = lambda image: cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert sharpness(decoded(out)[5]) >= 0.8 * sharpness(read_frame_at(VIDEO, 105)["image"])


def test_draw_gets_source_frame_numbers_and_marks_the_clip(tmp_path):
    seen = []

    def draw(image, frame_index):
        seen.append(frame_index)
        image[10:60, 10:60] = (255, 255, 255)

    out = tmp_path / "clip.mp4"
    write_clip(VIDEO, 30, 40, out, draw=draw)
    assert seen == list(range(30, 40))
    assert decoded(out)[3][20:50, 20:50].mean() > 240


def test_failed_clip_leaves_no_file(tmp_path):
    out = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError, match="wrote 10 of 20 frames"):
        write_clip(VIDEO, 230, 250, out)
    assert list(tmp_path.iterdir()) == []


def test_empty_clip_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Empty clip"):
        write_clip(VIDEO, 40, 40, tmp_path / "clip.mp4")


def test_clip_without_an_end_runs_to_the_end_of_the_video(tmp_path):
    out = tmp_path / "clip.mp4"
    assert write_clip(VIDEO, 230, None, out) == 10
    assert len(decoded(out)) == 10


def test_error_while_drawing_leaves_no_file_and_is_reported(tmp_path):
    def draw(image, frame_index):
        if frame_index == 5:
            raise ValueError("detector failed")

    with pytest.raises(ValueError, match="detector failed"):
        write_clip(VIDEO, 0, 20, tmp_path / "clip.mp4", draw=draw)
    assert list(tmp_path.iterdir()) == []


def test_video_of_odd_size_is_padded_to_even(tmp_path):
    source = tmp_path / "odd.avi"
    subprocess.run(
        [find_ffmpeg(), "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=641x361:rate=10:duration=1", "-c:v", "ffv1", str(source)],
        check=True,
    )
    out = tmp_path / "clip.mp4"
    assert write_clip(source, 0, None, out) == 10
    frames = decoded(out)
    assert len(frames) == 10 and frames[0].shape[:2] == (362, 642)
    # Compared on brightness, since this test pattern's sharp colours suffer from chroma subsampling.
    luma = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kept = luma(frames[3][:361, :641])
    original = luma(read_frame_at(source, 3)["image"])
    assert psnr(kept, original) >= 35
    assert psnr(kept, original) > psnr(kept[1:, 1:], original[:-1, :-1])  # The padding did not shift the image.


EVENT = {"event_id": "evt-test-f00100-t7", "trigger_frame": 100, "class_name": "car", "signal_source": "camera", "review_status": "pending"}


def rows(frames):
    return [[k, 10.0 + k, 20.0, 110.0 + k, 120.0, 2, 0.9] for k in frames]


def test_manifest_entry_maps_boxes_to_clip_frames(tmp_path):
    entry = cut_event_clip(VIDEO, EVENT, rows(range(60, 120)), 0, 240, tmp_path)
    assert (entry["source_start_frame"], entry["source_end_frame"]) == (28, 149)
    assert entry["frame_count"] == 121 and entry["trigger_frame_in_clip"] == 72
    first = entry["vehicle_boxes"][0]
    assert first["frame_in_clip"] == 60 - 28 and first["box_xyxy"] == [70.0, 20.0, 170.0, 120.0]
    assert len(entry["vehicle_boxes"]) == 60


def test_manifest_entry_drops_boxes_outside_the_clip(tmp_path):
    entry = cut_event_clip(VIDEO, EVENT, rows(range(60, 120)), 90, 110, tmp_path)
    assert [b["frame_in_clip"] for b in entry["vehicle_boxes"]] == list(range(20))
    assert all(0 <= b["frame_in_clip"] < entry["frame_count"] for b in entry["vehicle_boxes"])


def test_manifest_entry_leaves_the_plate_unread_and_the_event_pending(tmp_path):
    entry = cut_event_clip(VIDEO, EVENT, rows(range(90, 110)), 0, 240, tmp_path)
    assert entry["license_plate"] is None and entry["review_status"] == "pending"
    assert (tmp_path / entry["clip_file"]).is_file() and entry["clip_file"] == "evt-test-f00100-t7.mp4"
    json.dumps(entry)


def test_settings_change_the_range(tmp_path):
    settings = {**evidence.CLIP_SETTINGS, "pre_seconds": 1.0, "post_seconds": 1.0}
    entry = cut_event_clip(VIDEO, EVENT, rows(range(95, 110)), 0, 240, tmp_path, settings)
    assert (entry["source_start_frame"], entry["source_end_frame"]) == (76, 125)
