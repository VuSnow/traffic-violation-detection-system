"""Short video clips of candidate events: the input to plate reading and to human review.

A clip keeps the original pixels: it is cropped in time only, and nothing is drawn on it or
blurred. Compression is close to lossless, so that a plate stays as readable as in the source.
Each manifest entry says where the vehicle is in every frame of its clip.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from traffic_violation_detection.video import find_ffmpeg, iter_frames, read_video_metadata, video_writer

__all__ = ["CLIP_SETTINGS", "clip_frame_range", "cut_event_clip", "find_ffmpeg", "write_clip", "write_event_clips"]

CLIP_SETTINGS = {
    "pre_seconds": 3.0,       # at least this long before the trigger
    "max_pre_seconds": 10.0,  # a long track is followed back no further than this
    "post_seconds": 2.0,
    "crf": 14,                # H.264 quality: lower is better, and 14 is close to lossless
}


def clip_frame_range(trigger_frame: int, track_start: int, fps: float, limit_start: int, limit_end: int, settings: dict = CLIP_SETTINGS) -> tuple[int, int]:
    """Half-open frame range of an event's clip, inside ``[limit_start, limit_end)``.

    It reaches back to the start of the vehicle's track, because a vehicle seen from behind is
    nearest the camera, with its plate largest, before it reaches the line. The limits keep a
    clip from crossing a cut in the source video.
    """
    if not limit_start <= trigger_frame < limit_end:
        raise ValueError(f"Trigger frame {trigger_frame} is outside [{limit_start}, {limit_end})")
    wanted_start = min(track_start, trigger_frame - round(settings["pre_seconds"] * fps))
    earliest = trigger_frame - round(settings["max_pre_seconds"] * fps)
    start = max(limit_start, earliest, wanted_start)
    end = min(limit_end, trigger_frame + round(settings["post_seconds"] * fps) + 1)
    return start, end


def write_clip(
    video_path: str | Path,
    start: int,
    end: int | None,
    out_path: str | Path,
    crf: int = CLIP_SETTINGS["crf"],
    draw: Callable[[np.ndarray, int], None] | None = None,
) -> int:
    """Write frames ``[start, end)`` of a video as an H.264 file and return how many were written.

    With ``end=None`` it runs to the end of the video. ``draw(image, frame_index)``, if given, may
    draw on each frame before it is encoded. A failure leaves no partial clip behind.
    """
    if end is not None and end <= start:
        raise ValueError(f"Empty clip: start={start}, end={end}")
    fps = read_video_metadata(video_path)["fps"]
    written = 0
    with video_writer(out_path, fps, crf) as write_frame:
        for record in iter_frames(video_path, start, end):
            if draw is not None:
                draw(record["image"], record["frame_index"])
            write_frame(record["image"])
            written += 1
        if end is not None and written != end - start:
            raise RuntimeError(f"Clip {Path(out_path).name}: wrote {written} of {end - start} frames.")
    return written


def cut_event_clip(video_path: str | Path, event: dict, track_rows: list, limit_start: int, limit_end: int, out_dir: str | Path, settings: dict = CLIP_SETTINGS) -> dict:
    """Cut the clip of one candidate event and return its manifest entry.

    ``track_rows`` are the vehicle's rows, each starting with ``frame, x1, y1, x2, y2``.
    """
    meta = read_video_metadata(video_path)
    fps = meta["fps"]
    trigger = event["trigger_frame"]
    start, end = clip_frame_range(trigger, int(track_rows[0][0]), fps, limit_start, limit_end, settings)
    clip_name = f"{event['event_id']}.mp4"
    count = write_clip(video_path, start, end, Path(out_dir) / clip_name, crf=settings["crf"])
    return {
        "event_id": event["event_id"],
        "clip_file": clip_name,
        "source_start_frame": start,
        "source_end_frame": end,
        "frame_count": count,
        "fps": fps,
        "width": meta["width"],
        "height": meta["height"],
        "trigger_frame_in_clip": trigger - start,
        "vehicle_class": event["class_name"],
        "vehicle_boxes": [
            {"frame_in_clip": int(row[0]) - start, "box_xyxy": [float(v) for v in row[1:5]]}
            for row in track_rows
            if start <= row[0] < end
        ],
        "signal_source": event.get("signal_source"),
        "license_plate": None,
        "review_status": event["review_status"],
    }


def write_event_clips(
    video_path: str | Path,
    events: list[dict],
    track_rows_of: Callable[[dict], list],
    limits_of: Callable[[dict], tuple[int, int]],
    out_dir: str | Path,
    settings: dict = CLIP_SETTINGS,
) -> Path:
    """Cut one clip per event, write ``manifest.json`` beside them, and return its path. No events gives an empty manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [cut_event_clip(video_path, event, track_rows_of(event), *limits_of(event), out_dir, settings) for event in events]
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "video_id": Path(video_path).stem,
        "source_video": str(video_path),
        "settings": settings,
        "clips": entries,
    }, indent=1))
    return manifest
