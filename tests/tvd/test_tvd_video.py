import numpy as np
import pytest

from tvd.video import VideoReader, VideoWriter


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


def copy_frames(video, out, count: int = 10, crf: int = 14) -> None:
    with VideoReader(video) as reader, VideoWriter(out, reader.fps, crf) as writer:
        for frame in reader.frames(0, count):
            writer.write(frame.image)


def test_written_frames_are_read_back(tmp_path, sample_video):
    copy_frames(sample_video, tmp_path / "clip.mp4")
    with VideoReader(tmp_path / "clip.mp4") as reader, VideoReader(sample_video) as source:
        assert (reader.fps, reader.width, reader.height, sum(1 for _ in reader.frames())) == (24.0, 640, 360, 10)
        first, original = next(reader.frames(0, 1)), next(source.frames(0, 1))
    assert np.abs(first.image.astype(int) - original.image).mean() < 2  # CRF 14 is close to lossless.


def test_an_odd_size_gets_one_more_row_and_column(tmp_path):
    with VideoWriter(tmp_path / "odd.mp4", 24.0, 14) as writer:
        writer.write(np.zeros((21, 33, 3), np.uint8))
    with VideoReader(tmp_path / "odd.mp4") as reader:
        assert (reader.width, reader.height) == (34, 22)  # H.264 in yuv420p needs even sides.


def test_a_failure_leaves_no_file_behind(tmp_path):
    with pytest.raises(KeyError):
        with VideoWriter(tmp_path / "clip.mp4", 24.0, 14) as writer:
            writer.write(np.zeros((36, 64, 3), np.uint8))
            raise KeyError("stopped halfway")
    assert list(tmp_path.iterdir()) == []


def test_a_frame_of_another_size_is_refused(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        with VideoWriter(tmp_path / "clip.mp4", 24.0, 14) as writer:
            writer.write(np.zeros((36, 64, 3), np.uint8))
            writer.write(np.zeros((36, 66, 3), np.uint8))
    assert list(tmp_path.iterdir()) == []


def test_a_video_needs_frames(tmp_path):
    with pytest.raises(RuntimeError, match="No frames"):
        with VideoWriter(tmp_path / "clip.mp4", 24.0, 14):
            pass


def test_the_same_file_as_the_notebooks(tmp_path, sample_video):
    from traffic_violation_detection.video import iter_frames, video_writer

    copy_frames(sample_video, tmp_path / "tvd.mp4")
    with video_writer(tmp_path / "notebook.mp4", 24.0, 14) as write_frame:
        for record in iter_frames(sample_video, 0, 10):
            write_frame(record["image"])
    assert (tmp_path / "tvd.mp4").read_bytes() == (tmp_path / "notebook.mp4").read_bytes()
