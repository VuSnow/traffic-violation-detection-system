"""Tests for writing video in traffic_violation_detection.video."""

from pathlib import Path

import numpy as np
import pytest

from traffic_violation_detection.video import iter_frames, output_video_path, read_video_metadata, video_writer


def frame(value, size=(64, 48)):
    return np.full((size[1], size[0], 3), value, np.uint8)


def test_writer_keeps_every_frame_the_size_and_the_frame_rate(tmp_path):
    path = tmp_path / "out.mp4"
    with video_writer(path, 12.5, 14) as write_frame:
        for value in (10, 100, 200):
            write_frame(frame(value))
    meta = read_video_metadata(path)
    assert (meta["width"], meta["height"]) == (64, 48) and meta["fps"] == pytest.approx(12.5)
    assert [int(np.round(r["image"].mean())) for r in iter_frames(path)] == pytest.approx([10, 100, 200], abs=3)


def test_writer_creates_missing_folders(tmp_path):
    path = tmp_path / "a" / "b" / "out.mp4"
    with video_writer(path, 10, 14) as write_frame:
        write_frame(frame(50))
    assert path.is_file()


def test_nothing_appears_until_the_block_ends(tmp_path):
    path = tmp_path / "out.mp4"
    with video_writer(path, 10, 14) as write_frame:
        write_frame(frame(50))
        assert not path.exists()
    assert path.exists()


def test_error_in_the_block_leaves_no_file_and_keeps_the_error(tmp_path):
    with pytest.raises(ValueError, match="boom"):
        with video_writer(tmp_path / "out.mp4", 10, 14) as write_frame:
            write_frame(frame(50))
            raise ValueError("boom")
    assert list(tmp_path.iterdir()) == []


def test_error_keeps_an_earlier_output(tmp_path):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"earlier")
    with pytest.raises(ValueError):
        with video_writer(path, 10, 14) as write_frame:
            write_frame(frame(50))
            raise ValueError
    assert path.read_bytes() == b"earlier"


def test_no_frames_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="No frames"):
        with video_writer(tmp_path / "out.mp4", 10, 14):
            pass
    assert list(tmp_path.iterdir()) == []


def test_frames_of_another_size_are_an_error_not_a_garbled_video(tmp_path):
    with pytest.raises(ValueError, match="Frame of shape"):
        with video_writer(tmp_path / "out.mp4", 10, 14) as write_frame:
            write_frame(frame(50))
            write_frame(frame(50, size=(32, 24)))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [40, 100, 200])
def test_gray_stays_gray(tmp_path, value):
    # Guards against the conversion shift: without accurate_rnd, 100 came back as 95, 98, 96.
    path = tmp_path / "out.mp4"
    with video_writer(path, 10, 14) as write_frame:
        for _ in range(3):
            write_frame(frame(value))
    pixel = next(iter_frames(path))["image"][20, 30].astype(int)
    assert np.abs(pixel - value).max() <= 1 and pixel.max() - pixel.min() <= 1


def test_output_name_follows_the_video_name():
    assert output_video_path("data/2024-04-21 10-25-27.mp4", "out") == Path("out/2024-04-21 10-25-27_output.mp4")
    assert output_video_path(Path("x/video-test.avi"), Path("y")) == Path("y/video-test_output.mp4")
