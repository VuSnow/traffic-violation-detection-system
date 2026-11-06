import numpy as np
import pytest

from tvd.video import VideoReader


def test_the_reader_describes_the_video(sample_video):
    with VideoReader(sample_video) as reader:
        assert (reader.fps, reader.width, reader.height, reader.frame_count) == (24.0, 640, 360, 240)


def test_frames_of_a_range(sample_video):
    with VideoReader(sample_video) as reader:
        frames = list(reader.frames(10, 13))
    assert [f.index for f in frames] == [10, 11, 12]
    assert [f.timestamp_s for f in frames] == pytest.approx([10 / 24, 11 / 24, 12 / 24])
    assert frames[0].image.shape == (360, 640, 3)


def test_the_whole_video(sample_video):
    with VideoReader(sample_video) as reader:
        assert sum(1 for _ in reader.frames()) == 240


def test_the_same_pixels_as_the_notebooks(sample_video):
    from traffic_violation_detection.video import read_frame_at

    with VideoReader(sample_video) as reader:
        frame = next(reader.frames(100, 101))
    assert np.array_equal(frame.image, read_frame_at(sample_video, 100)["image"])


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")


def test_a_file_that_is_not_a_video_is_refused(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    with pytest.raises(RuntimeError, match="could not open"):
        VideoReader(path)
