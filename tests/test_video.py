"""Tests for traffic_violation_detection.video.

These are the Notebook 01 validation checks as automated tests. They decode the
test clips committed under data/ and compare every access path against
reference frame hashes from a plain sequential decode.
"""

import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_violation_detection.video import (
    frame_index_at_time,
    iter_frames,
    open_video,
    read_frame_at,
    read_frame_at_time,
    read_video_metadata,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_VIDEO = DATA_DIR / "video-test.mp4"  # H.264, 1920x1080, 59.94 FPS, 900 frames
SYNTHETIC_VIDEO = DATA_DIR / "traffic_sample.mp4"  # frame number and time burned in, 24 FPS
OVERHEAD_VIDEO = DATA_DIR / "car-detection.mp4"  # 12.5 FPS
ALL_VIDEOS = [SAMPLE_VIDEO, SYNTHETIC_VIDEO, OVERHEAD_VIDEO]


def frame_hash(image: np.ndarray) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


@pytest.fixture(scope="module", params=ALL_VIDEOS, ids=lambda p: p.name)
def video(request):
    path = request.param
    if not path.is_file():
        pytest.skip(f"test clip not available: {path}")
    return path


_reference_cache: dict[Path, list[str]] = {}


def reference_hashes(path: Path) -> list[str]:
    # Hashes of every frame from a plain sequential read() loop, independent of video.py.
    if path not in _reference_cache:
        hashes = []
        capture = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                hashes.append(frame_hash(image))
        finally:
            capture.release()
        _reference_cache[path] = hashes
    return _reference_cache[path]


# --- Sequential reading ---------------------------------------------------------


def test_decoded_count_matches_header(video):
    assert len(reference_hashes(video)) == read_video_metadata(video)["reported_frame_count"]


def test_iter_frames_returns_every_frame_once_in_order(video):
    meta = read_video_metadata(video)
    records = [(r["frame_index"], r["timestamp_s"], frame_hash(r["image"])) for r in iter_frames(video)]

    assert [i for i, _, _ in records] == list(range(meta["reported_frame_count"]))
    assert [h for _, _, h in records] == reference_hashes(video)
    timestamps = np.array([t for _, t, _ in records])
    assert np.all(np.diff(timestamps) > 0)
    # Constant frame rate: decoder timestamps equal index / fps.
    np.testing.assert_allclose(timestamps, np.arange(len(records)) / meta["fps"], atol=1e-6)


def test_frame_records_follow_the_frame_data_contract():
    record = next(iter_frames(SYNTHETIC_VIDEO))
    assert set(record) == {"video_id", "frame_index", "timestamp_s", "width", "height", "image"}
    assert record["video_id"] == "traffic_sample"
    assert record["image"].shape == (record["height"], record["width"], 3)
    assert record["image"].dtype == np.uint8


@pytest.mark.parametrize("start, stop, step", [(0, None, 5), (7, 50, 3), (100, None, 1), (239, None, 1)])
def test_iter_frames_ranges(start, stop, step):
    reference = reference_hashes(SYNTHETIC_VIDEO)
    end = len(reference) if stop is None else stop
    records = list(iter_frames(SYNTHETIC_VIDEO, start=start, stop=stop, step=step))

    assert [r["frame_index"] for r in records] == list(range(start, end, step))
    assert [frame_hash(r["image"]) for r in records] == reference[start:end:step]


def test_step_on_h264_sample_matches_reference():
    reference = reference_hashes(SAMPLE_VIDEO)
    records = list(iter_frames(SAMPLE_VIDEO, step=5))
    assert [frame_hash(r["image"]) for r in records] == reference[::5]


def test_stop_beyond_end_yields_remaining_frames_without_warning():
    n = len(reference_hashes(SYNTHETIC_VIDEO))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        records = list(iter_frames(SYNTHETIC_VIDEO, start=n - 3, stop=n + 50))
    assert [r["frame_index"] for r in records] == [n - 3, n - 2, n - 1]


@pytest.mark.parametrize("start, stop, step", [(-1, None, 1), (0, None, 0), (10, 5, 1)])
def test_iter_frames_rejects_invalid_ranges(start, stop, step):
    with pytest.raises(ValueError):
        next(iter_frames(SYNTHETIC_VIDEO, start=start, stop=stop, step=step))


# --- Random access --------------------------------------------------------------


def test_read_frame_at_returns_the_exact_frame(video):
    reference = reference_hashes(video)
    n = len(reference)
    rng = np.random.default_rng(42)
    targets = sorted(set([0, 1, n // 2, n - 1] + rng.integers(0, n, 8).tolist()))
    for k in targets:
        record = read_frame_at(video, k)
        assert record["frame_index"] == k
        assert frame_hash(record["image"]) == reference[k], f"seek to frame {k} returned a different frame"


@pytest.mark.parametrize("offset", [0, 99])
def test_read_frame_at_past_the_end_raises_index_error(offset):
    n = len(reference_hashes(SYNTHETIC_VIDEO))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # seeking far past the end also warns
        with pytest.raises(IndexError):
            read_frame_at(SYNTHETIC_VIDEO, n + offset)


@pytest.mark.parametrize(
    "t, fps, expected",
    [
        (0.0, 24, 0),
        (0.04, 24, 0),  # frame 0 is on screen until 1/24 s
        (1.0, 12.5, 12),  # frame 12 covers [0.96, 1.04); OpenCV's POS_MSEC seek returns 13
        (2.32, 12.5, 29),  # 2.32 * 12.5 = 28.999999999999996 in floating point
        (5.0, 59.94005994005994, 299),
    ],
)
def test_frame_index_at_time_uses_the_on_screen_rule(t, fps, expected):
    assert frame_index_at_time(t, fps) == expected


@pytest.mark.parametrize("fps", [10, 12.5, 24, 25, 30, 60000 / 1001, 60])
def test_frame_index_at_time_at_every_frame_start(fps):
    assert all(frame_index_at_time(i / fps, fps) == i for i in range(5000))


def test_frame_index_at_time_rejects_negative_time():
    with pytest.raises(ValueError):
        frame_index_at_time(-0.1, 24)


@pytest.mark.parametrize("t, expected_index", [(0.0, 0), (2.5, 60), (5.0, 120), (9.99, 239)])
def test_read_frame_at_time_matches_burned_in_frame_numbers(t, expected_index):
    # Expected indices were confirmed visually against the overlay in Notebook 01.
    record = read_frame_at_time(SYNTHETIC_VIDEO, t)
    assert record["frame_index"] == expected_index
    assert frame_hash(record["image"]) == reference_hashes(SYNTHETIC_VIDEO)[expected_index]


# --- Invalid inputs and cleanup ---------------------------------------------------


def test_missing_file_and_directory_raise_file_not_found(tmp_path):
    for path in (tmp_path / "does_not_exist.mp4", tmp_path):
        with pytest.raises(FileNotFoundError):
            read_video_metadata(path)


@pytest.mark.parametrize("content", [b"this is not a video", b""], ids=["text", "empty"])
def test_non_video_files_raise_runtime_error(tmp_path, content):
    path = tmp_path / "bad.mp4"
    path.write_bytes(content)
    with pytest.raises(RuntimeError):
        read_video_metadata(path)


def test_truncated_file_warns_about_missing_frames(tmp_path):
    data = SAMPLE_VIDEO.read_bytes()
    truncated = tmp_path / "truncated_half.mp4"
    truncated.write_bytes(data[: len(data) // 2])

    assert read_video_metadata(truncated)["reported_frame_count"] == 900  # header still claims all frames
    with pytest.warns(UserWarning, match="video ended at frame"):
        decoded = sum(1 for _ in iter_frames(truncated, step=50))
    assert 0 < decoded < 900 // 50


def test_open_video_releases_after_normal_exit_and_exception():
    with open_video(SYNTHETIC_VIDEO) as capture:
        held = capture
    assert not held.isOpened()

    with pytest.raises(ValueError):
        with open_video(SYNTHETIC_VIDEO) as capture:
            held = capture
            raise ValueError("simulated failure")
    assert not held.isOpened()
