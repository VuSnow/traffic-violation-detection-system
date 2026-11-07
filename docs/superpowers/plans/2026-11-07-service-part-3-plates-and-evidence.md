# Service Part 3: Plates and Evidence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn each candidate that Part 2 finds into a folder a person can review: its clip, three images (with the stop line and the signal state for red light), its plate read in a process of its own, and a JSON record of schema version 2, with a `review` command that writes it all.

**Architecture:** After Part 2's one pass, `review_check` cuts every clip in one more pass, reads each plate from its clip with models behind `BasePlateModels` (the real ones run in their own process, so ONNX Runtime never meets PyTorch), saves the evidence images with the overlay that each rule draws, and builds, de-duplicates and validates the records. The settings are in `configs/tvd/plates.yaml` and `configs/tvd/evidence.yaml`, and each record keeps them with the version of every settings file. Tests compare each step with what Notebooks 06, 07 and 08 saved, byte for byte where the notebooks saved files.

**Tech Stack:** Python 3.11, OpenCV 5.0.0, NumPy 2.4.6, ffmpeg (from `imageio-ffmpeg`), ONNX Runtime with `open-image-models` 0.6 and `fast-plate-ocr` 1.1 (the `plates` extra, installed for Notebook 06), pydantic 2, PyYAML 6, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-07-review-service-design.md` (Sections 2, 4, 5, 6, 9 and 12).

**Verified:** every file below, the state before each task's code (with the failure it shows) and the state after it were run on ai_server_2 before this plan was written. After the last task: 263 tests pass in `tests/tvd`; `review` on `traffic_video_modified.mp4` gives Notebook 08's two records, with the same clips and plate images byte for byte; and clip C of `rlr_occasions.mp4`, cut into its own file, gives Notebook 08's records for that clip.

## The five parts

| Part | Builds | Plan |
|---|---|---|
| 1. Foundations and perception | Data types, registry, video reader, detector, trackers, settings, `track` command | Done: `2026-11-07-service-part-1-foundations-and-perception.md` |
| 2. Scene and rules | Camera files, rule settings, camera motion, signal reading, the wrong-way and red-light rules, `check` command | Done: `2026-11-07-service-part-2-scene-and-rules.md` |
| **3. Plates and evidence** | Clips, plate models in a process of their own, voting, evidence images with the stop line and signal state, records of schema 2, `review` command | This document |
| 4. The service | Job queue, storage, REST API with FastAPI, around `check_video` and `review_check` | Later |
| 5. Deployment and MLOps | Docker, running on ai_server_2, logging, health checks, model versions | Later |

## What one `review` does

```text
video ──> check_video (Part 2): detect, track, every rule ──> findings, each with its vehicle's points; each rule's overlay
                │
                ├──> cut_clips: one more pass ──────────────> clips/<event id>.mp4           original pixels, CRF 14
                │
                ├──> for each clip: read_vehicle_plate ─────> plates/<event id>_f<frame>.png
                │        BasePlateModels ── pipe ──> onnx_process: detector + OCR (ONNX Runtime only here)
                │        12 frames ──> read_vehicle ──> vote: read | uncertain | unreadable | no_plate_found
                │
                ├──> write_evidence_images ──────────────────> evidence/<event id>_<role>.jpg  box; red light: stop line, signal state
                │
                └──> build_record ──> deduplicate ──> validate ──> records.json (schema 2, all pending), summary.json
```

A record is a candidate for a person to review, never a verdict. If plate reading fails, the record stays, and says `not_attempted`.

## Global Constraints

- Python 3.11, in the `tvd` environment: on ai_server_2, `~/workspace/dungvu/envs/tvd`. No new dependency in this part.
- Never let pip change OpenCV (5.0.0.93) or NumPy (2.4.6).
- New code goes in `src/tvd/`, and its tests in `tests/tvd/test_tvd_*.py`. The notebooks' package `traffic_violation_detection` does not change: it is the reference.
- **Nothing that a user might tune is written in code.** Plate settings are in `configs/tvd/plates.yaml`, and evidence settings in `configs/tvd/evidence.yaml`, with the values of Notebooks 06 and 07. Their models have no defaults.
- Notebook results are the reference. Where a test compares with them, the new code must match **exactly**: the same clip files and images byte for byte, the same plate results, the same record fields.
- **The service never confirms a violation.** Every record is `pending`, with no reviewer (spec, Section 1).
- **A clip keeps its original pixels.** Only evidence images are drawn on.
- **ONNX Runtime is never imported in the process that runs PyTorch** (Notebook 06).
- **Plate numbers are private.** `records.json` and `plates/` hold real plates: they stay in `outputs/`, which is git-ignored, on the server. Nothing prints a plate: the command prints counts, and a test that compares real plates asserts a boolean computed beforehand. Test doubles use made-up texts such as `TEST1234`, and `test_tvd_privacy.py` (Task 12) checks that no real plate of the private labels is in `src/tvd`, `tests/tvd`, `configs` or `docs`.
- Tests that need the GPU stack, the plate models, the videos in `data/`, or the saved outputs of Notebooks 05, 07 and 08 skip themselves where those are missing. The counts below are from the GPU server, where nothing skips.

## Review Focus

- **A plate step that fails**, because the models cannot start or their process dies, must not lose a violation: the record stays, with `not_attempted`, and the failure is in `summary.json`. `test_a_failed_plate_reading_keeps_every_record`, `test_without_plate_models_the_records_say_so` (Task 11), `test_review_keeps_the_records_when_the_plate_models_cannot_start` (Task 12).
- **An output folder that holds an earlier run** would mix two runs' images. A folder that is not new or empty, or a file in its place, is refused before any work: `test_a_folder_already_in_use_is_refused`, `test_a_file_where_the_folder_should_be_is_refused` (Task 11), `test_review_refuses_a_folder_in_use` (Task 12).
- **A `recorded_at` without a time zone** would put every absolute time hours off. It is refused before any model loads: `test_recorded_at_needs_a_time_zone` (Task 10), `test_review_needs_the_time_zone_of_the_first_frame_before_any_work` (Task 12).
- **A video that ends before its header says**, as a cut-off upload does, must not leave short or half-written clips: `test_no_clip_is_left_when_the_video_ends_too_soon` (Task 3), `test_a_failure_leaves_no_file_behind` (Task 1).
- **A record that looks decided, or points outside its folder,** must never reach a reviewer: `test_a_record_unfit_for_review_is_refused` (Task 10). `review_check` raises rather than save such a record (Task 11).

## How you will work

As in Parts 1 and 2: edit on your laptop, run on ai_server_2 with `scripts/server.sh`.

| Command | Does |
|---|---|
| `scripts/server.sh sync` | Makes the server's `src/`, `tests/`, `configs/`, `scripts/` and `pyproject.toml` the same as here |
| `scripts/server.sh test tests/tvd -q` | Runs pytest there |
| `scripts/server.sh run <command>` | Runs any command there, in the project folder, with the `tvd` environment |

Several tests read what the notebooks saved on the server, so that both implementations get the same input: Notebook 08's clips of `traffic_video_modified.mp4`, with their `manifest.json`, `plates.json`, plate images, evidence images and records, in `outputs/08_end_to_end_baseline/traffic_video_modified/`; Notebook 08's records and clips of `rlr_occasions.mp4`; and the wrong-way candidates and records of the time-reversed control of Notebooks 05 and 07.

## File structure

```text
src/tvd/
├── video.py                  Task 1    VideoWriter, find_ffmpeg                                   (changed)
├── rules/base.py             Task 2/9  Finding.points, Finding.as_json; RuleResult.overlay          (changed)
├── rules/wrong_way.py        Task 2    a finding keeps its track's points                         (changed)
├── rules/red_light.py        Task 2/9  a finding keeps its joined points; RedLightOverlay           (changed)
├── check.py                  Task 2/11 JSON without points; CheckResult.first_frame                 (changed)
├── drawing.py                Task 8/9  FrameOverlay, draw_vehicle_box, draw_label
├── evidence/
│   ├── __init__.py           Task 3/8  exports
│   ├── settings.py           Task 3    EvidenceSettings: clip, images, records
│   ├── clips.py              Task 3    clip_range, Clip, cut_clips
│   └── images.py             Task 8    evidence_frames, EvidenceImage, write_evidence_images
├── plates/
│   ├── __init__.py           Task 4-7  exports; importing it registers "onnx_worker"
│   ├── base.py               Task 4    PlateBox, PlateText, BasePlateModels, PLATE_MODELS, PAD
│   ├── settings.py           Task 4    PlatesConfig, ReadingSettings
│   ├── reading.py            Task 4    PlateRead, select_frames, crop_vehicle, pick_plate, read_vehicle
│   ├── voting.py             Task 5    PlateResult, vote, and the four statuses
│   ├── onnx.py               Task 6    OnnxPlateModels ("onnx_worker"): this process's side
│   ├── onnx_process.py       Task 6    the models' own process
│   └── vehicle.py            Task 7    read_vehicle_plate, save_plate_image
├── records.py                Task 10   RecordContext, build_record, validate_record, deduplicate_records
├── config.py                 Task 11   file_version                                               (changed)
├── factory.py                Task 6    build_plate_models                                         (changed)
├── review.py                 Task 11   review_check, ReviewResult, prepare_folder
└── cli.py                    Task 12   python -m tvd.cli review ...                               (changed)
configs/tvd/evidence.yaml     Task 3    clip, image and record settings
configs/tvd/plates.yaml       Task 4    the plate models, and how plates are read
configs/cameras/rlr_occasions-C.yaml   Task 12   clip C of rlr_occasions.mp4, as a camera of its own
docs/event_schema.md          Task 10   schema version 2                                           (changed)
tests/tvd/conftest.py         Task 4    FakePlateModels ("test-plate")                             (changed)
tests/tvd/notebook_data.py    Task 3-12 the notebooks' saved outputs, as tvd's inputs             (changed)
```

**How the parts depend on each other.** Arrows point at what a module uses. `cli.py` is the only place that builds concrete classes; `drawing.py` holds the overlay interface so that neither evidence nor the rules depend on each other.

```text
cli ──> factory ──> registry <── plates/onnx ("onnx_worker"), and Part 2's components
 ├──> check (Part 2) ──> rules ──> drawing
 └──> review ──> evidence ──> video, drawing, rules (Finding)
         ├──> plates/vehicle ──> plates/reading, plates/voting, video
         └──> records ──> evidence, plates, rules
plates/onnx ═══ pipe ═══ plates/onnx_process        ONNX Runtime is imported only there
```

## SOLID and patterns in this part

| Principle or pattern | Where |
|---|---|
| **S**ingle responsibility | `clips.py` cuts, `images.py` saves images, `reading.py` reads one frame, `voting.py` decides, `records.py` writes the record, `review.py` puts them in order, `cli.py` builds everything |
| **O**pen/closed | Another plate model is a new class with `@PLATE_MODELS.register`, chosen in `plates.yaml`. Another rule's drawing is a new `FrameOverlay` in its `RuleResult`. `review_check` does not change |
| **L**iskov substitution | `OnnxPlateModels` and `FakePlateModels` are interchangeable wherever `BasePlateModels` is used; so is any `FrameOverlay` |
| **I**nterface segregation | `BasePlateModels` has two methods and `close`; `FrameOverlay` has one |
| **D**ependency inversion | `review_check` receives built models, settings and a `RecordContext`; it never reads a file or builds a model |
| Proxy | `OnnxPlateModels` stands for models in another process |
| Strategy | Plate models by name; each rule's overlay |
| Context object | `RecordContext`: what every record of one video shares |

---

### Task 0: Branch, and the plate models on the server

**Why:** Part 3 reads plates with the two ONNX models of Notebook 06. They are already on the server; this task only checks that, and that Part 2 still passes, before anything changes.

**Files:** none.

- [ ] **Step 1: Start the branch from `main`, where Part 2 is merged**

```bash
git checkout main
git checkout -b service/part-3
```

- [ ] **Step 2: Check that Part 2 passes on the server**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `146 passed`.

- [ ] **Step 3: Check that the plate models are installed on the server**

They are the `plates` extra of `pyproject.toml`, installed for Notebook 06. Nothing is installed in this part.

```bash
scripts/server.sh run 'python -c "import fast_plate_ocr, open_image_models, onnxruntime; print(onnxruntime.get_available_providers())"'
```

Expected: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`. If an import fails, stop and ask: installing them must not change OpenCV or NumPy.

### Task 1: Writing video files

**Why:** Each candidate gets a clip, and Part 1 can only read video. `VideoWriter` sends frames to ffmpeg with the notebooks' own command, so the same frames give the same file, byte for byte: that is how this part proves its clips equal Notebook 08's. ffmpeg writes to a `.partial.mp4` name first, so a failure never leaves a broken clip behind.

**Files:**
- Change: `tests/tvd/test_tvd_video.py`
- Change: `src/tvd/video.py`

**Interfaces:**
- Consumes: `VideoReader` (Part 1). ffmpeg from the system, or from `imageio-ffmpeg`, a dependency since Phase 1.
- Produces: `VideoWriter(path, fps, crf)` with `write(image)`, `close()`, `abort()` and `frames_written`, usable in a `with` block (closed on success, aborted on an error); `find_ffmpeg() -> str`.

- [ ] **Step 1: Write the failing test**

In `tests/tvd/test_tvd_video.py`:

Replace:

```python
from tvd.video import VideoReader
```

with:

```python
from tvd.video import VideoReader, VideoWriter
```

Add to the end of the file:

```python
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
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_video.py -q
```

Expected: `E   ImportError: cannot import name 'VideoWriter' from 'tvd.video'`

- [ ] **Step 3: Write the code**

In `src/tvd/video.py`:

Replace:

```python
"""Reading video files."""

from __future__ import annotations
```

with:

```python
"""Reading and writing video files."""

from __future__ import annotations
```

Before:

```python
from collections.abc import Iterator
```

add:

```python
import shutil
import subprocess
```

After:

```python
from pathlib import Path

import cv2
```

add:

```python
import numpy as np
```

Add to the end of the file:

```python
def find_ffmpeg() -> str:
    """The system's ffmpeg, or else the one that the imageio-ffmpeg package brings."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Writing video needs ffmpeg: install it, or the imageio-ffmpeg package") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoWriter:
    """Writes BGR frames to an H.264 file with ffmpeg. Use it in a ``with`` block, or call ``close`` or ``abort``.

    The size comes from the first frame. The file gets its name only when ``close`` succeeds, so a failure
    never leaves a partial video behind. A lower ``crf`` is better quality: 14 is close to lossless.
    The ffmpeg command is the notebooks' own, so that the same frames give the same file.
    """

    def __init__(self, path: str | Path, fps: float, crf: int) -> None:
        self.path = Path(path)
        self.fps = fps
        self.crf = crf
        self.frames_written = 0
        self._partial = self.path.with_name(self.path.stem + ".partial.mp4")
        self._process: subprocess.Popen | None = None
        self._shape: tuple[int, ...] | None = None

    def _command(self, width: int, height: int) -> list[str]:
        return [
            find_ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
            # An odd side gets one black row or column: H.264 in yuv420p needs even sides.
            # Without accurate_rnd, the conversion from BGR shifts every pixel by a few levels.
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-sws_flags", "accurate_rnd",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self._partial),
        ]

    def write(self, image: np.ndarray) -> None:
        if self._process is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._shape = image.shape
            height, width = image.shape[:2]
            self._process = subprocess.Popen(self._command(width, height), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        elif image.shape != self._shape:
            # ffmpeg reads raw bytes: a frame of another size would garble the video without any error.
            raise ValueError(f"Frame of shape {image.shape} in a video of shape {self._shape}")
        try:
            self._process.stdin.write(image.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg stopped while writing {self.path.name}: {self._process.stderr.read().decode(errors='replace').strip()}") from error
        self.frames_written += 1

    def close(self) -> None:
        """Finish the file and give it its name."""
        if self._process is None:
            raise RuntimeError(f"No frames to write to {self.path.name}")
        _, errors = self._process.communicate()
        if self._process.returncode != 0:
            self._partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed on {self.path.name}: {errors.decode(errors='replace').strip()}")
        self._partial.replace(self.path)

    def abort(self) -> None:
        """Stop without a file."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._partial.unlink(missing_ok=True)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *rest: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `152 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_video.py src/tvd/video.py
git commit -m "Write videos with ffmpeg, as the notebooks do"
```

### Task 2: Each finding keeps its vehicle

**Why:** A clip needs the vehicle's box in each of its frames, to read its plate and to draw it. The rule already has those points, and only the red-light rule knows which tracks it joined into one vehicle, so the finding carries them. `as_json` leaves them out of the check's JSON, where a few hundred boxes per finding would only be noise.

**Files:**
- Change: `tests/tvd/test_tvd_check.py`
- Change: `tests/tvd/test_tvd_red_light.py`
- Change: `tests/tvd/test_tvd_rules.py`
- Change: `tests/tvd/test_tvd_wrong_way.py`
- Change: `src/tvd/check.py`
- Change: `src/tvd/rules/base.py`
- Change: `src/tvd/rules/red_light.py`
- Change: `src/tvd/rules/wrong_way.py`

**Interfaces:**
- Consumes: `Finding`, `RuleResult`, `WrongWayRule`, `RedLightRule`, `CheckResult` (Part 2); `TrackPoint` (Part 2, Task 1).
- Produces: `Finding.points: tuple[TrackPoint, ...]`, in frame pixels: for red light, the points of the joined track; `Finding.as_json() -> dict`, without the points. `CheckResult.as_json` uses it.

- [ ] **Step 1: Write the failing test**

The red-light test compares the points with the tracks that the notebook joins, on all 18 clips of Notebook 05b. The test of `check.py` passes already, before the code: it guards against the points reaching the JSON once they exist.

Add to the end of `tests/tvd/test_tvd_check.py`:

```python
def test_the_json_leaves_out_the_points_of_each_finding(sample_video):
    findings = [f for rule in run(sample_video).as_json()["rules"].values() for f in rule["findings"]]
    assert len(findings) == 2 and all("points" not in f for f in findings)
```

Add to the end of `tests/tvd/test_tvd_red_light.py`:

```python
@needs_stored_clips
def test_a_finding_keeps_the_points_of_the_track_as_the_notebooks_join_it():
    from traffic_violation_detection import red_light as notebook

    joined_findings = 0
    for clip_name in load_json(RED_LIGHT_SCENE)["clips"]:
        scene, clip, tracks, states = notebook_inputs(clip_name)
        joined = notebook.join_clip_tracks(tracks, None, notebook.RULE)
        section = red_light_section(scene, clip_name)
        rule = RedLightRule(section["scene"], {**SETTINGS, **section["settings"]})
        observation = Observation(history_from_rows(tracks).tracks, CameraMotion(), 30.0, 1280, 720, clip["start"], clip["end"])
        for finding in rule.judge(observation, SignalTimeline(states)).findings:
            assert [[p.frame_index, *p.box.as_list()] for p in finding.points] == [row[:5] for row in joined[finding.track_id]], clip_name
            joined_findings += "+" in finding.track_id
    assert joined_findings > 0  # Some candidates are joined tracks, so the joining is compared too.
```

In `tests/tvd/test_tvd_rules.py`:

Before:

```python
from tvd.motion import CameraMotion
```

add:

```python
from tvd.domain import Box
```

After:

```python
from tvd.rules import BaseRule, Finding, Observation, RuleResult
```

add:

```python
from tvd.tracking.history import TrackPoint
```

Add to the end of the file:

```python
def test_a_finding_keeps_its_vehicle_but_leaves_it_out_of_its_json():
    points = (TrackPoint(10, Box(0, 0, 10, 10), 0.9, "car"), TrackPoint(11, Box(1, 0, 11, 10), 0.9, "car"))
    finding = Finding("wrong_way", "7", "car", 10, 11, 11, "why", {"region_id": "road"}, points)
    assert finding.points[1].box == Box(1, 0, 11, 10)
    assert finding.as_json() == {
        "rule": "wrong_way", "track_id": "7", "class_name": "car", "start_frame": 10, "trigger_frame": 11, "end_frame": 11, "why": "why", "evidence": {"region_id": "road"},
    }
```

Add to the end of `tests/tvd/test_tvd_wrong_way.py`:

```python
def test_a_finding_keeps_every_point_of_its_track():
    (finding,) = judge(60, dy=3).findings
    assert [p.frame_index for p in finding.points] == list(range(60)) and finding.points[0].box == Box(300, 100, 340, 130)
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_check.py tests/tvd/test_tvd_red_light.py tests/tvd/test_tvd_rules.py tests/tvd/test_tvd_wrong_way.py -q
```

Expected: `3 failed, 37 passed`. The new tests of red light and wrong way stop with `AttributeError: 'Finding' object has no attribute 'points'`, and the one of `test_tvd_rules.py` with `TypeError: Finding.__init__() takes from 8 to 9 positional arguments but 10 were given`.

- [ ] **Step 3: Write the code**

In `src/tvd/check.py`:

Replace:

```python
from dataclasses import asdict, dataclass
```

with:

```python
from dataclasses import dataclass
```

Replace:

```python
                name: {"settings": r.settings, "verdicts": r.verdict_counts(), "findings": [asdict(f) for f in r.findings], "records": r.records}
```

with:

```python
                name: {"settings": r.settings, "verdicts": r.verdict_counts(), "findings": [f.as_json() for f in r.findings], "records": r.records}
```

In `src/tvd/rules/base.py`:

Replace:

```python
from dataclasses import dataclass, field
```

with:

```python
from dataclasses import dataclass, field, fields
```

Replace:

```python
from tvd.tracking.history import Track
```

with:

```python
from tvd.tracking.history import Track, TrackPoint
```

After:

```python
    ``start_frame`` to ``end_frame`` is the stretch of video that shows it; ``trigger_frame`` is the moment the rule fired.
```

add:

```python
    ``points`` are the vehicle in every frame of its track, in frame pixels: where clips and plate reading find it.
```

After:

```python
    evidence: dict[str, Any] = field(default_factory=dict)
```

add:

```python
    points: tuple[TrackPoint, ...] = ()

    def as_json(self) -> dict[str, Any]:
        """Everything but the points: they are many, and the evidence keeps the vehicle's boxes in its clip."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "points"}
```

In `src/tvd/rules/red_light.py`:

After:

```python
                "signal_source": self.scene.signal_source,
            },
```

add:

```python
            points=tuple(points),
```

In `src/tvd/rules/wrong_way.py`:

Replace:

```python
from tvd.tracking.history import Track
```

with:

```python
from tvd.tracking.history import Track, TrackPoint
```

Replace:

```python
            findings += [self._finding(path, group) for group in groups]
```

with:

```python
            findings += [self._finding(path, group, track.points) for group in groups]
```

Replace:

```python
    def _finding(self, path: TrackPath, group: list[dict]) -> Finding:
```

with:

```python
    def _finding(self, path: TrackPath, group: list[dict], points: list[TrackPoint]) -> Finding:
```

After:

```python
                "evidence_frames": [start, trigger, end],
            },
```

add:

```python
            points=tuple(points),
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `156 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_check.py tests/tvd/test_tvd_red_light.py tests/tvd/test_tvd_rules.py tests/tvd/test_tvd_wrong_way.py src/tvd/check.py src/tvd/rules/base.py src/tvd/rules/red_light.py src/tvd/rules/wrong_way.py
git commit -m "Keep each finding's vehicle points"
```

### Task 3: Evidence settings and clips

**Why:** A reviewer watches each candidate's clip, and its plate is read from it. Every setting of the evidence is in `configs/tvd/evidence.yaml`, with the values of Notebook 07. **Efficiency:** decoding is most of the work, so all clips are written during one pass over the video, each by its own ffmpeg process, instead of decoding the video again for each clip.

**Files:**
- Change: `tests/tvd/notebook_data.py`
- Create: `tests/tvd/test_tvd_clips.py`
- Create: `configs/tvd/evidence.yaml`
- Create: `src/tvd/evidence/settings.py`
- Create: `src/tvd/evidence/clips.py`
- Create: `src/tvd/evidence/__init__.py`

**Interfaces:**
- Consumes: `VideoReader`, `VideoWriter` (Task 1); `Finding.points` (Task 2); `StrictModel`, `load_config` (Part 1).
- Produces: `EvidenceSettings(version, clip: ClipSettings, images: ImageSettings, records: RecordSettings)`, `load_evidence_settings(path)`; `clip_range(trigger_frame, track_start, fps, first_frame, end_frame, settings) -> (start, end)`; `Clip(path, start, end, fps, width, height, trigger_frame, boxes)` where `boxes` maps a frame of the clip (from 0) to the vehicle's `Box`, with `frame_count`, `trigger_in_clip` and `box_near(frame_in_clip, tolerance)`; `cut_clips(reader, findings: Mapping[str, Finding], first_frame, end_frame, folder, settings) -> dict[str, Clip]`.

- [ ] **Step 1: Write the failing test**

`notebook_data.py` gains what the comparison with Notebook 08 needs. `finding_from_clip` rebuilds a finding from a clip that Notebook 08 cut, so that the same clip can be cut again and compared byte for byte.

Add to the end of `tests/tvd/notebook_data.py`:

```python
MODIFIED_VIDEO = ROOT / "data" / "traffic_video_modified.mp4"
NOTEBOOK_08 = OUTPUTS / "08_end_to_end_baseline" / "traffic_video_modified"
NOTEBOOK_08_CLIPS = NOTEBOOK_08 / "violations" / "traffic_video_modified_violation_clips"


def notebook_08_clips() -> list[dict]:
    """The manifest entries of the two red-light clips that Notebook 08 cut from traffic_video_modified.mp4."""
    return load_json(NOTEBOOK_08_CLIPS / "manifest.json")["clips"]


def finding_from_clip(entry: dict):
    """A red-light finding with the vehicle boxes and trigger of a Notebook 08 clip: enough to cut the same clip again."""
    from tvd.rules import Finding
    from tvd.tracking.history import TrackPoint

    start = entry["source_start_frame"]
    points = tuple(TrackPoint(start + b["frame_in_clip"], Box(*b["box_xyxy"]), 0.9, entry["vehicle_class"]) for b in entry["vehicle_boxes"])
    trigger = start + entry["trigger_frame_in_clip"]
    return Finding("red_light", entry["event_id"].rsplit("-t", 1)[1], entry["vehicle_class"], points[0].frame_index, trigger, points[-1].frame_index, "", {}, points)
```

`tests/tvd/test_tvd_clips.py`:

```python
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
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_clips.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.evidence'`

- [ ] **Step 3: Write the code**

`configs/tvd/evidence.yaml`:

```yaml
# How each candidate's evidence is cut and kept, for every camera. Every record stores these settings.
version: 1
clip:                     # original pixels, cut in time only
  pre_seconds: 3.0        # at least this long before the trigger
  max_pre_seconds: 10.0   # a long track is followed back no further than this
  post_seconds: 2.0
  crf: 14                 # H.264 quality, lower is better: 14 is close to lossless, so a plate stays as readable as in the source
images:                   # three images per candidate: before, at and after the trigger
  offset_seconds: 0.7     # how far the images before and after are from the trigger
  jpeg_quality: 90
  box_tolerance_frames: 3 # the vehicle's box is drawn if it was tracked this close to the image's frame
records:
  duplicate_seconds: 5.0  # two records of one rule, this close in time, of one track or one read plate, are one occurrence
```

`src/tvd/evidence/settings.py`:

```python
"""How each candidate's evidence is cut and kept: ``configs/tvd/evidence.yaml``. There are no defaults in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tvd.config import StrictModel, load_config


class ClipSettings(StrictModel):
    pre_seconds: float = Field(ge=0)
    max_pre_seconds: float = Field(ge=0)
    post_seconds: float = Field(ge=0)
    crf: int = Field(ge=0, le=51)


class ImageSettings(StrictModel):
    offset_seconds: float = Field(gt=0)
    jpeg_quality: int = Field(ge=1, le=100)
    box_tolerance_frames: int = Field(ge=0)


class RecordSettings(StrictModel):
    duplicate_seconds: float = Field(ge=0)


class EvidenceSettings(StrictModel):
    version: int
    clip: ClipSettings
    images: ImageSettings
    records: RecordSettings


def load_evidence_settings(path: str | Path) -> EvidenceSettings:
    return load_config(path, EvidenceSettings)
```

`src/tvd/evidence/clips.py`:

```python
"""Short clips of candidates, with original pixels: what a reviewer watches, and where plates are read.

A clip is cut in time only: nothing is drawn on it, and it is compressed close to losslessly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tvd.domain import Box
from tvd.evidence.settings import ClipSettings
from tvd.rules import Finding
from tvd.video import VideoReader, VideoWriter


def clip_range(trigger_frame: int, track_start: int, fps: float, first_frame: int, end_frame: int, settings: ClipSettings) -> tuple[int, int]:
    """The frames of a candidate's clip, from the first to one past the last, inside the analysed frames ``[first_frame, end_frame)``.

    It reaches back to the start of the vehicle's track, because a vehicle seen from behind is nearest the
    camera, with its plate largest, before it reaches the line; but no further than ``max_pre_seconds``.
    """
    if not first_frame <= trigger_frame < end_frame:
        raise ValueError(f"Trigger frame {trigger_frame} is outside the analysed frames [{first_frame}, {end_frame})")
    wanted_start = min(track_start, trigger_frame - round(settings.pre_seconds * fps))
    earliest = trigger_frame - round(settings.max_pre_seconds * fps)
    start = max(first_frame, earliest, wanted_start)
    end = min(end_frame, trigger_frame + round(settings.post_seconds * fps) + 1)
    return start, end


@dataclass(frozen=True)
class Clip:
    """Frames ``start`` to ``end`` (not included) of the source video, saved at ``path``.

    ``boxes`` are the vehicle's boxes by frame of the clip, which counts from 0: only frames where it was tracked have one.
    """

    path: Path
    start: int
    end: int
    fps: float
    width: int
    height: int
    trigger_frame: int
    boxes: dict[int, Box]

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    @property
    def trigger_in_clip(self) -> int:
        return self.trigger_frame - self.start

    def box_near(self, frame_in_clip: int, tolerance: int) -> Box | None:
        """The vehicle's box at a frame of the clip, or at the nearest frame within ``tolerance``; None if it was not tracked there."""
        nearest = min(self.boxes, key=lambda k: abs(k - frame_in_clip), default=None)
        return self.boxes[nearest] if nearest is not None and abs(nearest - frame_in_clip) <= tolerance else None


def cut_clips(
    reader: VideoReader, findings: Mapping[str, Finding], first_frame: int, end_frame: int, folder: str | Path, settings: ClipSettings,
) -> dict[str, Clip]:
    """Cut one clip per finding, named by its key, in a single pass over the video.

    Decoding is most of the work, so every clip is written while the frames go by once. If anything fails,
    no clip is left behind.
    """
    folder = Path(folder)
    ranges = {
        name: clip_range(f.trigger_frame, f.points[0].frame_index if f.points else f.start_frame, reader.fps, first_frame, end_frame, settings)
        for name, f in findings.items()
    }
    if not ranges:
        return {}
    writers = {name: VideoWriter(folder / f"{name}.mp4", reader.fps, settings.crf) for name in ranges}
    try:
        for frame in reader.frames(min(s for s, _ in ranges.values()), max(e for _, e in ranges.values())):
            for name, (start, end) in ranges.items():
                if start <= frame.index < end:
                    writers[name].write(frame.image)
                    if frame.index == end - 1:
                        writers[name].close()
        short = {name: writers[name].frames_written for name, (start, end) in ranges.items() if writers[name].frames_written != end - start}
        if short:
            raise RuntimeError(f"The video ended before these clips were complete (frames written): {short}")
    except BaseException:
        for name, writer in writers.items():
            writer.abort()
            writer.path.unlink(missing_ok=True)
        raise
    return {
        name: Clip(
            path=writers[name].path, start=start, end=end, fps=reader.fps, width=reader.width, height=reader.height,
            trigger_frame=findings[name].trigger_frame,
            boxes={p.frame_index - start: p.box for p in findings[name].points if start <= p.frame_index < end},
        )
        for name, (start, end) in ranges.items()
    }
```

`src/tvd/evidence/__init__.py`:

```python
"""The evidence of each candidate: its clip, and images from it."""

from tvd.evidence.clips import Clip, clip_range, cut_clips
from tvd.evidence.settings import ClipSettings, EvidenceSettings, ImageSettings, RecordSettings, load_evidence_settings

__all__ = ["Clip", "ClipSettings", "EvidenceSettings", "ImageSettings", "RecordSettings", "clip_range", "cut_clips", "load_evidence_settings"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `167 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/notebook_data.py tests/tvd/test_tvd_clips.py configs/tvd/evidence.yaml src/tvd/evidence/settings.py src/tvd/evidence/clips.py src/tvd/evidence/__init__.py
git commit -m "Cut each candidate's clip in one pass"
```

### Task 4: Plate models, and reading one frame

**Why:** Plate reading needs two models, a detector and an OCR, behind one small interface: `BasePlateModels`, with a registry like every other component. The reading of one frame is ported from Notebook 06: crop the vehicle with a margin, keep the plate nearest the middle of the bottom of its box (a roof sign is not its plate), read it, and say why a read cannot be used. A fake implementation in `conftest.py` makes all of it testable without a GPU, and a test runs the notebook's own code on the same fake, so the port is compared exactly.

**Files:**
- Change: `tests/tvd/conftest.py`
- Create: `tests/tvd/test_tvd_plates.py`
- Create: `configs/tvd/plates.yaml`
- Create: `src/tvd/plates/base.py`
- Create: `src/tvd/plates/settings.py`
- Create: `src/tvd/plates/reading.py`
- Create: `src/tvd/plates/__init__.py`

**Interfaces:**
- Consumes: `Box` (Part 1); `Registry` (Part 1); `ComponentConfig`, `StrictModel`, `load_config` (Part 1).
- Produces: `PAD`; `PlateBox(box, confidence)`; `PlateText(characters, probabilities)`; `BasePlateModels` with `detect(image) -> list[PlateBox]`, `recognize(plate) -> PlateText`, `close()`, usable in a `with` block; `PLATE_MODELS`; `PlatesConfig(version, models, reading: ReadingSettings)`, `load_plates_config(path)`; `PlateRead(frame, text, mean_prob, min_prob, has_gap, detection_confidence, plate_width, plate_height, plate_box, rejected, image)` with `as_json()`; `select_frames(boxes, count, min_gap) -> list[int]`; `read_vehicle(image, frame, box, models, settings) -> PlateRead | None`. The test double `FakePlateModels`, registered as `test-plate`.

- [ ] **Step 1: Write the failing test**

Images are BGR everywhere in tvd. The real OCR wants RGB: that conversion belongs to its implementation (Task 6), not to the interface. The fake reads `TEST1234`, an obviously made-up text: tests never hold a real plate.

In `tests/tvd/conftest.py`:

After:

```python
from tvd.domain import Box, Detection, Frame, TrackedObject
```

add:

```python
from tvd.plates import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
```

Before:

```python
@pytest.fixture
def sample_video() -> Path:
```

add:

```python
@PLATE_MODELS.register("test-plate")
class FakePlateModels(BasePlateModels):
    """Finds one plate in the lower middle of every image, a quarter of its width, and reads ``text`` on it with confidence ``prob``."""

    def __init__(self, text: str = "TEST1234", prob: float = 0.99) -> None:
        self.text = text
        self.prob = prob
        self.closed = False

    def detect(self, image) -> list[PlateBox]:
        height, width = image.shape[:2]
        return [PlateBox(Box(0.375 * width, 0.8 * height, 0.625 * width, 0.9 * height), 0.9)]

    def recognize(self, plate) -> PlateText:
        characters = self.text.ljust(9, PAD)
        return PlateText(characters, tuple(self.prob if c != PAD else 1.0 for c in characters))

    def close(self) -> None:
        self.closed = True
```

`tests/tvd/test_tvd_plates.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, FakePlateModels

from tvd.config import ConfigError
from tvd.domain import Box
from tvd.plates import PLATE_MODELS, BasePlateModels, PlateBox, load_plates_config, read_vehicle, select_frames
from tvd.plates.reading import pick_plate

PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
READING = PLATES.reading
IMAGE = np.zeros((360, 640, 3), np.uint8)
CAR = Box(200, 100, 400, 260)


class NotebookModels:
    """Models with the calls that the notebooks' plate code makes, answering as the given tvd models do."""

    def __init__(self, models: BasePlateModels) -> None:
        self.models = models

    def predict(self, image):
        return [SimpleNamespace(confidence=p.confidence, bounding_box=SimpleNamespace(x1=p.box.x1, y1=p.box.y1, x2=p.box.x2, y2=p.box.y2)) for p in self.models.detect(image)]

    def run_one(self, image, return_confidence=False, remove_pad_char=True):
        text = self.models.recognize(image)
        return SimpleNamespace(plate=text.characters, char_probs=np.array(text.probabilities))


def test_the_settings_file_has_the_notebooks_values():
    from traffic_violation_detection.plates import PLATE_SETTINGS

    assert {"version": PLATES.version, **PLATES.models.options, **READING.model_dump()} == PLATE_SETTINGS and PLATES.models.type == "onnx_worker"


def test_a_wrong_setting_is_named(tmp_path):
    path = tmp_path / "plates.yaml"
    path.write_text((ROOT / "configs" / "tvd" / "plates.yaml").read_text().replace("min_agreement: 0.67", "min_agreement: 67"))
    with pytest.raises(ConfigError, match="reading.min_agreement: Input should be less than or equal to 1"):
        load_plates_config(path)


def test_models_are_chosen_by_name_and_closed_after_use():
    with PLATE_MODELS.create("test-plate", text="FAKE9999") as models:
        assert models.recognize(IMAGE).characters == "FAKE9999_"
    assert models.closed


def test_plate_models_must_detect_and_recognize():
    class DetectorOnly(BasePlateModels):
        def detect(self, image):
            return []

    with pytest.raises(TypeError):
        DetectorOnly()


def test_a_plate_is_read_where_the_vehicle_is():
    read = read_vehicle(IMAGE, 7, CAR, FakePlateModels(), READING)
    assert (read.frame, read.text, read.mean_prob, read.min_prob, read.has_gap, read.rejected) == (7, "TEST1234", 0.99, 0.99, False, None)
    assert read.plate_box == Box(267, 242, 332, 263) and (read.plate_width, read.plate_height) == (65, 21) and read.image.shape == (21, 65, 3)


@pytest.mark.parametrize("models, reason", [
    (FakePlateModels(prob=0.9), "low_confidence"),
    (FakePlateModels(text="TES"), "too_short"),
    (FakePlateModels(text="TES_T1234"), "gap_in_text"),
])
def test_a_read_that_cannot_be_used_says_why(models, reason):
    assert read_vehicle(IMAGE, 0, CAR, models, READING).rejected == reason


def test_a_plate_narrower_than_30_px_is_too_small():
    assert read_vehicle(IMAGE, 0, Box(300, 200, 380, 260), FakePlateModels(), READING).rejected == "too_small"


def test_a_sign_high_on_the_vehicle_is_not_its_plate():
    class RoofSign(FakePlateModels):
        def detect(self, image):
            height, width = image.shape[:2]
            return [PlateBox(Box(0.4 * width, 0.1 * height, 0.6 * width, 0.2 * height), 0.99)]

    assert read_vehicle(IMAGE, 0, CAR, RoofSign(), READING) is None


def test_the_plate_nearest_the_middle_of_the_bottom_edge_wins():
    middle, side = PlateBox(Box(280, 240, 320, 260), 0.5), PlateBox(Box(210, 240, 250, 260), 0.99)
    assert pick_plate([side, middle], CAR, (0, 0), 0.4) is middle


def test_the_frames_where_the_vehicle_is_tallest_and_apart():
    boxes = {k: Box(0, 0, 10, height) for k, height in enumerate([10, 50, 49, 48, 20, 47, 46])}
    assert select_frames(boxes, 3, 2) == [1, 3, 5]


@pytest.mark.parametrize("text, prob", [("TEST1234", 0.99), ("TES", 0.99), ("TES_T1234", 0.97), ("FAKE9999", 0.9)])
@pytest.mark.parametrize("box", [Box(200, 100, 400, 260), Box(300, 200, 380, 260), Box(-5, 300, 120, 359.5), Box(600, 0, 640, 30)])
def test_the_same_reads_as_the_notebooks(text, prob, box):
    from traffic_violation_detection.plates import PLATE_SETTINGS, read_vehicle_in_frame

    image = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    models = FakePlateModels(text, prob)
    ours = read_vehicle(image, 3, box, models, READING)
    theirs = read_vehicle_in_frame(image, [3, *box.as_list()], NotebookModels(models), NotebookModels(models), PLATE_SETTINGS)
    assert (ours is None) == (theirs is None)
    if ours is not None:
        assert ours.as_json() == {k: v for k, v in theirs.items() if k != "plate_image"} and np.array_equal(ours.image, theirs["plate_image"])
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_plates.py -q
```

Expected: `ImportError while loading conftest ...`, then `E   ModuleNotFoundError: No module named 'tvd.plates'`. Every test of the package fails this way until the code exists, because `conftest.py` imports it.

- [ ] **Step 3: Write the code**

`configs/tvd/plates.yaml`:

```yaml
# How plates are read, for every camera. Every record stores these settings. The values are those of Notebook 06.
version: 1
models:                       # what finds and reads plates; change "type" to swap the implementation
  type: onnx_worker           # both models in a process of their own, on the GPU when there is one
  options:
    detector_model: yolo-v9-t-640-license-plate-end2end
    ocr_model: cct-s-v2-global-model
    cpu_threads: 8            # only for what ONNX runs on the CPU, such as on a machine with no GPU
reading:
  frames_per_track: 12        # the frames where the vehicle looks largest
  min_frame_gap: 4            # neighbouring frames are nearly the same, so they would not be independent votes
  vehicle_margin: 0.15        # a box can cut off the plate, so the crop extends beyond it
  lowest_plate_center: 0.4    # a plate's centre is below this fraction of the box height: this rejects roof signs
  min_plate_width: 30         # px; narrower plates are too small to read
  min_chars: 4
  min_char_prob: 0.95         # the least sure character of a usable read; a wrong plate is worse than none, so this is strict
  min_votes: 2                # frames that must agree for "read"
  min_agreement: 0.67         # the share of usable frames that must agree: 2 of 3 is not enough, and 3 of 4 is
```

`src/tvd/plates/base.py`:

```python
"""What plate reading needs from its models: to find plates in an image, and to read the characters of one."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from tvd.domain import Box
from tvd.registry import Registry

PAD = "_"
"""What an OCR model puts in a character slot that holds no character."""


@dataclass(frozen=True)
class PlateBox:
    """A plate found in an image, in that image's pixels."""

    box: Box
    confidence: float


@dataclass(frozen=True)
class PlateText:
    """What the OCR read on one plate, slot by slot: ``characters`` holds ``PAD`` in empty slots, and each slot has a confidence."""

    characters: str
    probabilities: tuple[float, ...]


class BasePlateModels(ABC):
    """A plate detector and an OCR model, loaded once and used for every vehicle. Images are BGR, as OpenCV gives them.

    Close them when done, or use them in a ``with`` block: an implementation may hold a process or a GPU.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[PlateBox]:
        """The plates in an image."""

    @abstractmethod
    def recognize(self, plate: np.ndarray) -> PlateText:
        """The characters on one plate's image."""

    def close(self) -> None:
        """Release what the models hold. Nothing, by default."""

    def __enter__(self) -> BasePlateModels:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


PLATE_MODELS: Registry[BasePlateModels] = Registry("plate models")
```

`src/tvd/plates/settings.py`:

```python
"""How plates are read: ``configs/tvd/plates.yaml``. There are no defaults in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tvd.config import ComponentConfig, StrictModel, load_config


class ReadingSettings(StrictModel):
    frames_per_track: int = Field(ge=1)
    min_frame_gap: int = Field(ge=0)
    vehicle_margin: float = Field(ge=0)
    lowest_plate_center: float = Field(ge=0, le=1)
    min_plate_width: int = Field(ge=1)
    min_chars: int = Field(ge=1)
    min_char_prob: float = Field(ge=0, le=1)
    min_votes: int = Field(ge=1)
    min_agreement: float = Field(ge=0, le=1)


class PlatesConfig(StrictModel):
    version: int
    models: ComponentConfig
    reading: ReadingSettings


def load_plates_config(path: str | Path) -> PlatesConfig:
    return load_config(path, PlatesConfig)
```

`src/tvd/plates/reading.py`:

```python
"""Reading one vehicle's plate in one frame, as Notebook 06 does. One frame is never enough: see ``voting.py``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tvd.domain import Box
from tvd.plates.base import PAD, BasePlateModels, PlateBox
from tvd.plates.settings import ReadingSettings


@dataclass(frozen=True)
class PlateRead:
    """One frame's reading of one vehicle's plate. ``rejected`` says why it cannot be used, or is None if it can."""

    frame: int
    text: str
    mean_prob: float
    min_prob: float
    has_gap: bool
    detection_confidence: float
    plate_width: int
    plate_height: int
    plate_box: Box
    rejected: str | None
    image: np.ndarray = field(repr=False, compare=False)

    def as_json(self) -> dict[str, Any]:
        """The read without its image, with the keys of the notebooks' reads."""
        return {
            "text": self.text, "mean_prob": self.mean_prob, "min_prob": self.min_prob, "has_gap": self.has_gap, "frame": self.frame,
            "detection_confidence": self.detection_confidence, "plate_width": self.plate_width, "plate_height": self.plate_height,
            "plate_box_xyxy": self.plate_box.as_list(), "rejected": self.rejected,
        }


def select_frames(boxes: dict[int, Box], count: int, min_gap: int) -> list[int]:
    """The frames where the vehicle looks tallest, at least ``min_gap`` frames apart. ``boxes`` are by frame, in order."""
    chosen: list[int] = []
    for frame, _ in sorted(boxes.items(), key=lambda item: item[1].height, reverse=True):
        if all(abs(frame - c) >= min_gap for c in chosen):
            chosen.append(frame)
        if len(chosen) == count:
            break
    return chosen


def crop_vehicle(image: np.ndarray, box: Box, margin: float) -> tuple[np.ndarray, tuple[int, int]]:
    """The vehicle's region with a margin around its box, and where that region starts in the image."""
    margin_x, margin_y = margin * box.width, margin * box.height
    height, width = image.shape[:2]
    left, top = max(0, int(box.x1 - margin_x)), max(0, int(box.y1 - margin_y))
    right, bottom = min(width, int(box.x2 + margin_x)), min(height, int(box.y2 + margin_y))
    return image[top:bottom, left:right], (left, top)


def pick_plate(plates: list[PlateBox], box: Box, offset: tuple[int, int], lowest_center: float) -> PlateBox | None:
    """The plate most likely to be this vehicle's: inside its box, and nearest the middle of its bottom edge.

    A box can also hold a neighbour's plate, a roof sign or a shop sign, so the most confident plate is not enough.
    """
    width, height = max(box.width, 1), max(box.height, 1)
    best, best_distance = None, float("inf")
    for plate in plates:
        center_x, center_y = plate.box.center
        across = (center_x + offset[0] - box.x1) / width
        down = (center_y + offset[1] - box.y1) / height
        if not (0 <= across <= 1 and lowest_center <= down <= 1.05):
            continue
        distance = float(np.hypot(across - 0.5, down - 1.0))
        if distance < best_distance:
            best, best_distance = plate, distance
    return best


def read_text(models: BasePlateModels, plate: np.ndarray) -> tuple[str, float, float, bool]:
    """The text on a plate image, its mean and lowest character confidence, and whether an empty slot splits it.

    Confidence counts real characters only, because the OCR is also sure of its padding.
    """
    result = models.recognize(plate)
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    real = [i for i, char in enumerate(result.characters) if char != PAD]
    text = "".join(result.characters[i] for i in real)
    mean_prob = round(float(probabilities[real].mean()), 4) if real else 0.0
    min_prob = round(float(probabilities[real].min()), 4) if real else 0.0
    return text, mean_prob, min_prob, PAD in result.characters.rstrip(PAD)


def rejection_reason(text: str, min_prob: float, has_gap: bool, plate_width: int, settings: ReadingSettings) -> str | None:
    """Why a read cannot be used, or None if it can."""
    if plate_width < settings.min_plate_width:
        return "too_small"
    if len(text) < settings.min_chars:
        return "too_short"
    if has_gap:
        return "gap_in_text"
    if min_prob < settings.min_char_prob:
        return "low_confidence"
    return None


def read_vehicle(image: np.ndarray, frame: int, box: Box, models: BasePlateModels, settings: ReadingSettings) -> PlateRead | None:
    """Read one vehicle's plate in one frame. None means that no plate of this vehicle was found there."""
    crop, offset = crop_vehicle(image, box, settings.vehicle_margin)
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    plate = pick_plate(models.detect(crop), box, offset, settings.lowest_plate_center)
    if plate is None:
        return None
    b = plate.box
    plate_image = crop[max(0, int(b.y1)):max(0, int(b.y2)), max(0, int(b.x1)):max(0, int(b.x2))]
    if plate_image.size == 0:
        return None
    text, mean_prob, min_prob, has_gap = read_text(models, plate_image)
    width, height = int(plate_image.shape[1]), int(plate_image.shape[0])
    return PlateRead(
        frame=frame, text=text, mean_prob=mean_prob, min_prob=min_prob, has_gap=has_gap,
        detection_confidence=round(float(plate.confidence), 4), plate_width=width, plate_height=height,
        plate_box=Box(int(b.x1) + offset[0], int(b.y1) + offset[1], int(b.x2) + offset[0], int(b.y2) + offset[1]),
        rejected=rejection_reason(text, min_prob, has_gap, width, settings), image=plate_image.copy(),
    )
```

`src/tvd/plates/__init__.py`:

```python
"""Plate reading: models behind an interface, the reading of one frame, and the vote over many frames."""

from tvd.plates.base import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.plates.reading import PlateRead, read_vehicle, select_frames
from tvd.plates.settings import PlatesConfig, ReadingSettings, load_plates_config

__all__ = [
    "PAD", "PLATE_MODELS", "BasePlateModels", "PlateBox", "PlateRead", "PlateText", "PlatesConfig", "ReadingSettings",
    "load_plates_config", "read_vehicle", "select_frames",
]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `195 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/conftest.py tests/tvd/test_tvd_plates.py configs/tvd/plates.yaml src/tvd/plates/base.py src/tvd/plates/settings.py src/tvd/plates/reading.py src/tvd/plates/__init__.py
git commit -m "Read a vehicle's plate in one frame"
```

### Task 5: Voting over many frames

**Why:** One frame is never enough: the OCR can be sure of a wrong text. The frames vote, and the result has one of four statuses that say how far to trust it (Notebook 06). `uncertain` is a suggestion and must never look like `read`. The vote is a pure function, so it is compared with the notebook's on 300 random sets of reads, and on the reads that Notebook 08 saved.

**Files:**
- Change: `tests/tvd/notebook_data.py`
- Create: `tests/tvd/test_tvd_voting.py`
- Change: `src/tvd/plates/__init__.py`
- Create: `src/tvd/plates/voting.py`

**Interfaces:**
- Consumes: `PlateRead`, `ReadingSettings` (Task 4).
- Produces: `READ`, `UNCERTAIN`, `UNREADABLE`, `NO_PLATE`; `PlateResult(status, text, votes, usable_reads, plates_found, agreement, mean_prob, other_texts, reads)` with `best_read()` (the read kept as evidence) and `as_json()`; `vote(reads, settings) -> PlateResult`.

- [ ] **Step 1: Write the failing test**

**Plate numbers stay private.** Notebook 08's results hold real plates. A test that compares them computes the comparison first, `matches = ours == theirs`, and asserts `matches`: had it asserted `ours == theirs`, a failure would print both plates.

Add to the end of `tests/tvd/notebook_data.py`:

```python
NOTEBOOK_08_PLATES = NOTEBOOK_08 / "plates"


def notebook_08_plates() -> dict[str, dict]:
    """Notebook 08's plate result for each clip, by event id. They hold real plate numbers: never print them."""
    return {e["event_id"]: e for e in load_json(NOTEBOOK_08_PLATES / "plates.json")["events"]}


def plate_read_from_json(read: dict):
    """A read that the notebooks saved. Its image was not saved: it is replaced by one black pixel."""
    import numpy as np

    from tvd.plates import PlateRead

    keys = ("frame", "text", "mean_prob", "min_prob", "has_gap", "detection_confidence", "plate_width", "plate_height")
    return PlateRead(*(read[k] for k in keys), Box(*read["plate_box_xyxy"]), read["rejected"], np.zeros((1, 1, 3), np.uint8))
```

`tests/tvd/test_tvd_voting.py`:

```python
import numpy as np
import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_PLATES, notebook_08_plates, plate_read_from_json

from tvd.domain import Box
from tvd.plates import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateRead, load_plates_config, vote

READING = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml").reading
needs_notebook_08 = pytest.mark.skipif(not NOTEBOOK_08_PLATES.is_dir(), reason="Needs the outputs of Notebook 08, as on the GPU server.")


def plate_read(text: str, frame: int = 0, mean_prob: float = 0.99, rejected: str | None = None, width: int = 60) -> PlateRead:
    return PlateRead(frame, text, mean_prob, mean_prob, False, 0.9, width, 20, Box(0, 0, width, 20), rejected, np.zeros((20, width, 3), np.uint8))


def test_three_frames_that_agree_are_a_read():
    result = vote([plate_read("TEST1234", k) for k in range(3)], READING)
    assert (result.status, result.text, result.votes, result.usable_reads, result.agreement, result.mean_prob) == (READ, "TEST1234", 3, 3, 1.0, 0.99)


def test_two_of_three_frames_are_only_a_suggestion():
    result = vote([plate_read("TEST1234"), plate_read("TEST1235"), plate_read("TEST1234")], READING)
    assert (result.status, result.agreement, result.other_texts) == (UNCERTAIN, 0.667, {"TEST1235": 1})


def test_three_of_four_frames_are_a_read():
    assert vote([plate_read("TEST1234")] * 3 + [plate_read("TEST1235")], READING).status == READ


def test_one_usable_frame_is_only_a_suggestion():
    assert vote([plate_read("TEST1234"), plate_read("TES", rejected="too_short")], READING).status == UNCERTAIN


def test_a_plate_with_no_usable_read_is_unreadable():
    result = vote([plate_read("TES", rejected="too_short")], READING)
    assert (result.status, result.text, result.plates_found) == (UNREADABLE, None, 1)


def test_no_plate_found():
    result = vote([], READING)
    assert (result.status, result.text, result.best_read()) == (NO_PLATE, None, None)


def test_a_tie_goes_to_the_text_read_with_more_confidence():
    reads = [plate_read("TEST1234", mean_prob=0.96), plate_read("TEST1235", mean_prob=0.99)]
    assert vote(reads, READING).text == "TEST1235"


def test_the_evidence_is_the_largest_plate_of_the_chosen_text():
    reads = [plate_read("TEST1234", 1, width=50), plate_read("TEST1235", 2, width=90), plate_read("TEST1234", 3, width=70)]
    assert vote(reads, READING).best_read().frame == 3


def test_the_same_votes_as_the_notebooks():
    from traffic_violation_detection.plates import PLATE_SETTINGS, best_read, combine_reads

    rng = np.random.default_rng(1)
    for _ in range(300):
        reads = [
            plate_read(str(rng.choice(["TEST1234", "TEST1235", "FAKE9999"])), k, round(float(rng.uniform(0.9, 1.0)), 4),
                       rng.choice([None, None, None, "too_short", "low_confidence"]), int(rng.integers(30, 90)))
            for k in range(int(rng.integers(0, 8)))
        ]
        theirs = combine_reads([r.as_json() for r in reads], PLATE_SETTINGS)
        ours = vote(reads, READING)
        assert {k: v for k, v in ours.as_json().items() if k != "reads"} == theirs
        expected = best_read([r.as_json() for r in reads], theirs["text"])
        assert (ours.best_read().frame if ours.best_read() else None) == (expected["frame"] if expected else None)


@needs_notebook_08
def test_the_same_plates_as_notebook_08_from_its_reads():
    for event_id, saved in notebook_08_plates().items():
        expected = saved["license_plate"]
        result = vote([plate_read_from_json(r) for r in expected["reads"]], READING)
        matches = result.as_json() == expected  # Compared outside the assert, so that a failure never prints a real plate.
        assert matches, f"{event_id} differs from Notebook 08"
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_voting.py -q
```

Expected: `E   ImportError: cannot import name 'NO_PLATE' from 'tvd.plates'`

- [ ] **Step 3: Write the code**

`src/tvd/plates/__init__.py`, replaced as a whole:

```python
"""Plate reading: models behind an interface, the reading of one frame, and the vote over many frames."""

from tvd.plates.base import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.plates.reading import PlateRead, read_vehicle, select_frames
from tvd.plates.settings import PlatesConfig, ReadingSettings, load_plates_config
from tvd.plates.voting import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateResult, vote

__all__ = [
    "NO_PLATE", "PAD", "PLATE_MODELS", "READ", "UNCERTAIN", "UNREADABLE", "BasePlateModels", "PlateBox", "PlateRead", "PlateResult", "PlateText",
    "PlatesConfig", "ReadingSettings", "load_plates_config", "read_vehicle", "select_frames", "vote",
]
```

`src/tvd/plates/voting.py`:

```python
"""One plate per vehicle, from the reads of many frames, as Notebook 06 does.

One frame's confidence is not enough: the OCR can be sure of a wrong text, and of an empty one. So the
frames vote, and each vehicle ends in one explicit status:

- ``read``: enough frames agree on one text;
- ``uncertain``: there is a text, but too few frames agree, so it is only a suggestion;
- ``unreadable``: a plate was found, but no frame gave a usable text;
- ``no_plate_found``: no plate of this vehicle was found.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tvd.plates.reading import PlateRead
from tvd.plates.settings import ReadingSettings

READ = "read"
UNCERTAIN = "uncertain"
UNREADABLE = "unreadable"
NO_PLATE = "no_plate_found"


@dataclass(frozen=True)
class PlateResult:
    status: str
    text: str | None
    votes: int
    usable_reads: int
    plates_found: int
    agreement: float
    mean_prob: float
    other_texts: dict[str, int]
    reads: tuple[PlateRead, ...] = field(default=(), repr=False)

    def best_read(self) -> PlateRead | None:
        """The read to keep as evidence: of the chosen text if there is one, and with the largest plate."""
        pool = [r for r in self.reads if r.text == self.text and r.rejected is None] or list(self.reads)
        return max(pool, key=lambda r: r.plate_width * r.plate_height, default=None)

    def as_json(self) -> dict[str, Any]:
        return {
            "status": self.status, "text": self.text, "votes": self.votes, "usable_reads": self.usable_reads, "plates_found": self.plates_found,
            "agreement": self.agreement, "mean_prob": self.mean_prob, "other_texts": dict(self.other_texts), "reads": [r.as_json() for r in self.reads],
        }


def vote(reads: list[PlateRead], settings: ReadingSettings) -> PlateResult:
    """Combine one vehicle's reads. On a tie, the text with more total confidence wins."""
    usable = [r for r in reads if r.rejected is None]
    if not reads:
        return PlateResult(NO_PLATE, None, 0, 0, 0, 0.0, 0.0, {}, ())
    if not usable:
        return PlateResult(UNREADABLE, None, 0, 0, len(reads), 0.0, 0.0, {}, tuple(reads))
    votes = Counter(r.text for r in usable)
    text = max(votes, key=lambda t: (votes[t], sum(r.mean_prob for r in usable if r.text == t)))
    agreement = votes[text] / len(usable)
    agreed = votes[text] >= settings.min_votes and agreement >= settings.min_agreement
    return PlateResult(
        status=READ if agreed else UNCERTAIN,
        text=text,
        votes=votes[text],
        usable_reads=len(usable),
        plates_found=len(reads),
        agreement=round(agreement, 3),
        mean_prob=round(float(np.mean([r.mean_prob for r in usable if r.text == text])), 4),
        other_texts={t: n for t, n in votes.items() if t != text},
        reads=tuple(reads),
    )
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `205 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/notebook_data.py tests/tvd/test_tvd_voting.py src/tvd/plates/__init__.py src/tvd/plates/voting.py
git commit -m "Vote on a plate over many frames"
```

### Task 6: The plate models in a process of their own

**Why:** Notebook 06 found that ONNX Runtime, once it has run on the GPU, moves PyTorch's detections by about 0.01 px in the same process: 15 of 136 tracks changed. So the models run in a process of their own, and only that process imports ONNX Runtime. `OnnxPlateModels` is a **proxy**: it offers `detect` and `recognize` here, and sends each call through a pipe to the other process. An error there is raised here, and the process stays up for the next call.

**Files:**
- Change: `tests/tvd/test_tvd_factory.py`
- Create: `tests/tvd/test_tvd_onnx_plates.py`
- Create: `src/tvd/plates/onnx_process.py`
- Create: `src/tvd/plates/onnx.py`
- Change: `src/tvd/plates/__init__.py`
- Change: `src/tvd/factory.py`

**Interfaces:**
- Consumes: `BasePlateModels`, `PlateBox`, `PlateText`, `PLATE_MODELS` (Task 4); `build` and `ComponentConfig` (Part 1).
- Produces: `OnnxPlateModels(detector_model, ocr_model, cpu_threads)`, registered as `onnx_worker`, with `providers`, `pid` and `call(method, *arguments)`; `build_plate_models(config) -> BasePlateModels` in `factory.py`. The process: `python -m tvd.plates.onnx_process '<options as JSON>'`.

- [ ] **Step 1: Write the failing test**

These tests start the real models, so they skip where the `plates` extra is not installed. One of them compares answers with the notebooks' own worker, on frames and plate images of Notebook 08.

In `tests/tvd/test_tvd_factory.py`:

Replace:

```python
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
```

with:

```python
from tvd.factory import build_detector, build_motion_estimator, build_plate_models, build_rules, build_tracker
```

Add to the end of the file:

```python
def test_plate_models_are_built_by_their_name():
    with build_plate_models(ComponentConfig(type="test-plate", options={"text": "FAKE9999"})) as models:
        assert models.text == "FAKE9999"


def test_a_misspelt_plate_model_option_is_reported():
    with pytest.raises(ConfigError, match="onnx_worker: .*unexpected keyword argument 'ocr'"):
        build_plate_models(ComponentConfig(type="onnx_worker", options={"ocr": "cct-s-v2-global-model"}))
```

`tests/tvd/test_tvd_onnx_plates.py`:

```python
"""The real plate models, in their own process. These tests skip where the models are not installed."""

import gc
import importlib.util
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_CLIPS, NOTEBOOK_08_PLATES

from tvd.factory import build_plate_models
from tvd.plates import load_plates_config
from tvd.video import VideoReader

PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
GRAY = np.full((60, 120, 3), 128, np.uint8)
pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("onnxruntime", "fast_plate_ocr", "open_image_models")), reason="The plate models are not installed here.",
)
needs_notebook_08 = pytest.mark.skipif(not NOTEBOOK_08_PLATES.is_dir(), reason="Needs the outputs of Notebook 08, as on the GPU server.")


@pytest.fixture(scope="module")
def models():
    with build_plate_models(PLATES.models) as started:
        yield started


def test_the_models_run_in_another_process(models):
    assert models.pid != os.getpid()


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="There is no GPU here.")
def test_the_models_use_the_gpu_where_there_is_one(models):
    assert models.providers[0] == "CUDAExecutionProvider"


def test_a_gray_image_has_no_plate_and_ten_empty_slots(models):
    text = models.recognize(GRAY)
    assert models.detect(GRAY) == [] and text.characters == "_" * 10 and len(text.probabilities) == 10


def test_a_large_image_crosses_between_the_processes(models):
    assert models.detect(np.zeros((1080, 1920, 3), np.uint8)) == []


def test_an_error_there_is_raised_here_and_the_models_go_on(models):
    with pytest.raises(RuntimeError, match="Plate models: "):
        models.call("detect", "not an image")
    with pytest.raises(RuntimeError, match="KeyError"):
        models.call("unknown")
    assert models.detect(GRAY) == []


def test_an_unknown_model_fails_when_the_process_starts():
    with pytest.raises(RuntimeError, match="Plate models: "):
        build_plate_models(PLATES.models.model_copy(update={"options": {**PLATES.models.options, "ocr_model": "no-such-model"}}))


def test_closing_stops_the_process():
    started = build_plate_models(PLATES.models)
    process = started._process
    started.close()
    assert process.poll() is not None


def test_the_process_stops_when_nothing_refers_to_the_models():
    started = build_plate_models(PLATES.models)
    process = started._process
    del started
    gc.collect()
    assert process.wait(timeout=15) == 0


def test_a_stopped_process_gives_a_clear_error():
    started = build_plate_models(PLATES.models)
    started._process.kill()
    started._process.wait()
    with pytest.raises(RuntimeError, match="process stopped"):
        started.detect(GRAY)


def test_onnx_stays_out_of_the_callers_process():
    # In one process with PyTorch it changes the detections, and with them the tracks.
    code = (
        "import sys\n"
        "from tvd.factory import build_plate_models\n"
        "from tvd.plates import load_plates_config\n"
        "models = build_plate_models(load_plates_config('configs/tvd/plates.yaml').models)\n"
        "assert models.detect(__import__('numpy').zeros((60, 120, 3), 'uint8')) == []\n"
        "assert 'onnxruntime' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, cwd=ROOT)


@needs_notebook_08
def test_the_same_answers_as_the_notebooks_models(models):
    from traffic_violation_detection.plate_worker import PlateDetector, PlateReader, PlateWorker
    from traffic_violation_detection.plates import PLATE_SETTINGS

    with VideoReader(next(NOTEBOOK_08_CLIPS.glob("*.mp4"))) as clip:
        frames = [f.image for f in clip.frames()][::12]
    crops = [cv2.imread(str(p)) for p in sorted((NOTEBOOK_08_PLATES / "plate_crops").glob("*.png"))]
    worker = PlateWorker(PLATE_SETTINGS)
    try:
        detector, reader = PlateDetector(worker), PlateReader(worker)
        ours = [[(p.box.as_list(), p.confidence) for p in models.detect(image)] for image in frames]
        theirs = [[([d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2], d.confidence) for d in detector.predict(image)] for image in frames]
        our_texts = [models.recognize(crop) for crop in crops]
        their_texts = [reader.run_one(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_confidence=True, remove_pad_char=False) for crop in crops]
    finally:
        worker.close()
    assert ours == theirs and sum(map(len, ours)) > 0
    same_texts = [(t.characters, t.probabilities) for t in our_texts] == [(t.plate, tuple(float(p) for p in t.char_probs)) for t in their_texts]
    assert same_texts and len(crops) == 2  # Compared outside the assert, so that a failure never prints a real plate.
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_factory.py tests/tvd/test_tvd_onnx_plates.py -q
```

Expected: Both files fail to import: `E   ImportError: cannot import name 'build_plate_models' from 'tvd.factory'`.

- [ ] **Step 3: Write the code**

Two files, one per process. `onnx_process.py` is never imported by `tvd.plates`: run as `python -m tvd.plates.onnx_process`, a module that the package had imported already would run twice, and register `onnx_worker` twice.

`src/tvd/plates/onnx_process.py`:

```python
"""The process that runs the real plate models: ``python -m tvd.plates.onnx_process '<options as JSON>'``.

It answers requests from its standard input until the caller closes it. Requests and answers are pickled
``(method, arguments)`` and ``(ok, value)``. Only this process ever imports ONNX Runtime: see ``onnx.py``.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any

import cv2

from tvd.domain import Box
from tvd.plates.base import PlateBox, PlateText


def create_models(detector_model: str, ocr_model: str, cpu_threads: int) -> tuple[Any, Any]:
    """The plate detector of open-image-models and the OCR of fast-plate-ocr, on the GPU when there is one."""
    try:
        import onnxruntime as ort
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector
    except ImportError as error:
        raise RuntimeError('Plate reading needs its models: pip install -e ".[plates]"') from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
        # ONNX Runtime needs the CUDA libraries that pip installs with PyTorch, which are not on the loader's path.
        # Without them, on a machine whose system CUDA is another version, it runs on the CPU with only a warning.
        ort.preload_dlls()
    arguments = {"providers": providers, "sess_options": options}
    return create_detector(detector_model, **arguments), LicensePlateRecognizer(ocr_model, **arguments)


def detect(detector: Any, image) -> list[PlateBox]:
    return [PlateBox(Box(d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2), float(d.confidence)) for d in detector.predict(image)]


def recognize(reader: Any, plate) -> PlateText:
    # The OCR takes an array to be RGB already: it converts only image files.
    prediction = reader.run_one(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), return_confidence=True, remove_pad_char=False)
    return PlateText(prediction.plate, tuple(float(p) for p in prediction.char_probs))


def main() -> None:
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # What the libraries print must not reach the replies.

    def reply(ok: bool, value: Any) -> None:
        pickle.dump((ok, value), replies, protocol=pickle.HIGHEST_PROTOCOL)
        replies.flush()

    try:
        detector, reader = create_models(**json.loads(sys.argv[1]))
    except Exception as error:
        reply(False, f"{type(error).__name__}: {error}")
        return
    reply(True, detector.model.get_providers())
    methods = {"detect": lambda image: detect(detector, image), "recognize": lambda plate: recognize(reader, plate)}
    while True:
        try:
            method, arguments = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            reply(True, methods[method](*arguments))
        except Exception as error:
            reply(False, f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
```

`src/tvd/plates/onnx.py`:

```python
"""The real plate models, in a process of their own: ``onnx_worker``.

ONNX Runtime and PyTorch disturb each other inside one process: after ONNX had run on the GPU, PyTorch's
detections moved by about 0.01 px, enough to change 15 of 136 tracks, so a video's result depended on
what had run before it (Notebook 06). With ONNX in another process, the detections stay the same.
Why it happens inside one process is not known.

This class is the caller's side; ``onnx_process.py`` is the other process.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from tvd.plates.base import PLATE_MODELS, BasePlateModels, PlateBox, PlateText

STOP_TIMEOUT_SECONDS = 10


def stop_process(process: subprocess.Popen) -> None:
    """Close the process's input, so that it ends by itself, and force it if it does not."""
    if process.poll() is None:
        try:
            process.stdin.close()
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()


@PLATE_MODELS.register("onnx_worker")
class OnnxPlateModels(BasePlateModels):
    """Starts the models' process, and asks it to detect and read. It stops when closed, or when nothing refers to it any more."""

    def __init__(self, detector_model: str, ocr_model: str, cpu_threads: int) -> None:
        options = {"detector_model": detector_model, "ocr_model": ocr_model, "cpu_threads": cpu_threads}
        source_root = str(Path(__file__).resolve().parents[2])  # So that the process runs this same code.
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")]))}
        self._process = subprocess.Popen(
            [sys.executable, "-m", "tvd.plates.onnx_process", json.dumps(options)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment,
        )
        self._finalizer = weakref.finalize(self, stop_process, self._process)
        self.providers: list[str] = self._receive()

    @property
    def pid(self) -> int:
        return self._process.pid

    def _receive(self) -> Any:
        try:
            ok, value = pickle.load(self._process.stdout)
        except EOFError as error:
            raise RuntimeError(f"The plate models' process stopped (exit code {self._process.wait()}); its messages are on standard error.") from error
        if not ok:
            raise RuntimeError(f"Plate models: {value}")
        return value

    def call(self, method: str, *arguments: Any) -> Any:
        """Run a method in the process and return its answer. An error there is raised here, and the process goes on."""
        try:
            pickle.dump((method, arguments), self._process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise RuntimeError(f"The plate models' process stopped (exit code {self._process.poll()}).") from error
        return self._receive()

    def detect(self, image: np.ndarray) -> list[PlateBox]:
        return self.call("detect", image)

    def recognize(self, plate: np.ndarray) -> PlateText:
        return self.call("recognize", plate)

    def close(self) -> None:
        self._finalizer()
```

`src/tvd/plates/__init__.py`, replaced as a whole:

```python
"""Plate reading: models behind an interface, the reading of one frame, and the vote over many frames.

Importing this package registers every implementation of the models in it.
"""

from tvd.plates import onnx  # noqa: F401  Registers "onnx_worker".
from tvd.plates.base import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.plates.reading import PlateRead, read_vehicle, select_frames
from tvd.plates.settings import PlatesConfig, ReadingSettings, load_plates_config
from tvd.plates.voting import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateResult, vote

__all__ = [
    "NO_PLATE", "PAD", "PLATE_MODELS", "READ", "UNCERTAIN", "UNREADABLE", "BasePlateModels", "PlateBox", "PlateRead", "PlateResult", "PlateText",
    "PlatesConfig", "ReadingSettings", "load_plates_config", "read_vehicle", "select_frames", "vote",
]
```

In `src/tvd/factory.py`:

After:

```python
from tvd.motion import MOTION_ESTIMATORS, BaseMotionEstimator
```

add:

```python
from tvd.plates import PLATE_MODELS, BasePlateModels
```

Before:

```python
def build_rules(camera: CameraConfig, defaults: dict[str, dict[str, Any]]) -> list[BaseRule]:
```

add:

```python
def build_plate_models(config: ComponentConfig) -> BasePlateModels:
    return build(PLATE_MODELS, config)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `218 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_factory.py tests/tvd/test_tvd_onnx_plates.py src/tvd/plates/onnx_process.py src/tvd/plates/onnx.py src/tvd/plates/__init__.py src/tvd/factory.py
git commit -m "Run the plate models in a process of their own"
```

### Task 7: A vehicle's plate, from its clip

**Why:** This joins Tasks 4 and 5: read the frames where the vehicle looks largest, and vote. As in Notebook 08, the plate is read from the candidate's clip, which is what the reviewer sees; a clip is close to lossless, so nothing is lost. The image of the best read is saved as a PNG, without more compression.

**Files:**
- Create: `tests/tvd/test_tvd_vehicle_plate.py`
- Change: `src/tvd/plates/__init__.py`
- Create: `src/tvd/plates/vehicle.py`

**Interfaces:**
- Consumes: `VideoReader` (Part 1); `select_frames`, `read_vehicle` (Task 4); `vote`, `PlateResult` (Task 5); `BasePlateModels` (Task 4).
- Produces: `read_vehicle_plate(video, boxes: dict[int, Box], models, settings) -> PlateResult`; `save_plate_image(result, folder, name) -> Path | None`, saving `<name>_f<frame>.png`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_vehicle_plate.py`:

```python
import importlib.util

import cv2
import pytest
from conftest import ROOT, FakePlateModels
from notebook_data import NOTEBOOK_08_CLIPS, NOTEBOOK_08_PLATES, notebook_08_clips, notebook_08_plates

from tvd.domain import Box
from tvd.factory import build_plate_models
from tvd.plates import NO_PLATE, READ, load_plates_config, read_vehicle_plate, save_plate_image
from tvd.video import VideoReader

PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
READING = PLATES.reading
needs_models_and_notebook_08 = pytest.mark.skipif(
    not all(importlib.util.find_spec(n) for n in ("onnxruntime", "fast_plate_ocr", "open_image_models")) or not NOTEBOOK_08_PLATES.is_dir(),
    reason="Needs the plate models and the outputs of Notebook 08, as on the GPU server.",
)


def growing_car(frames: range) -> dict[int, Box]:
    """A car whose box grows by 1 px per frame, as one coming toward the camera."""
    return {k: Box(200, 100, 400 + k, 260 + k) for k in frames}


def test_the_frames_where_the_vehicle_is_largest_vote(sample_video):
    with VideoReader(sample_video) as video:
        result = read_vehicle_plate(video, growing_car(range(40, 100)), FakePlateModels(), READING)
    assert (result.status, result.text, result.votes) == (READ, "TEST1234", 12)
    assert [r.frame for r in result.reads] == list(range(55, 100, 4))  # The 12 tallest boxes, 4 frames apart, in the order of the video.


def test_a_vehicle_never_tracked_has_no_plate(sample_video):
    with VideoReader(sample_video) as video:
        assert read_vehicle_plate(video, {}, FakePlateModels(), READING).status == NO_PLATE


def test_the_best_read_is_saved_as_a_png(tmp_path, sample_video):
    with VideoReader(sample_video) as video:
        result = read_vehicle_plate(video, growing_car(range(40, 100)), FakePlateModels(), READING)
    path = save_plate_image(result, tmp_path / "plates", "evt-1")
    assert path.name == "evt-1_f00099.png" and cv2.imread(str(path)).shape == (result.best_read().plate_height, result.best_read().plate_width, 3)


def test_no_read_saves_nothing(tmp_path, sample_video):
    with VideoReader(sample_video) as video:
        assert save_plate_image(read_vehicle_plate(video, {}, FakePlateModels(), READING), tmp_path, "evt-1") is None


@needs_models_and_notebook_08
def test_the_same_plates_and_images_as_notebook_08(tmp_path):
    saved = notebook_08_plates()
    with build_plate_models(PLATES.models) as models:
        for entry in notebook_08_clips():
            boxes = {b["frame_in_clip"]: Box(*b["box_xyxy"]) for b in entry["vehicle_boxes"]}
            with VideoReader(NOTEBOOK_08_CLIPS / entry["clip_file"]) as clip:
                result = read_vehicle_plate(clip, boxes, models, READING)
            path = save_plate_image(result, tmp_path, entry["event_id"])
            expected = saved[entry["event_id"]]
            same_result = result.as_json() == expected["license_plate"]  # Outside the assert, so that a failure never prints a real plate.
            assert same_result and result.status == READ, entry["event_id"]
            assert path.name == expected["plate_crop"] and path.read_bytes() == (NOTEBOOK_08_PLATES / "plate_crops" / expected["plate_crop"]).read_bytes()
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_vehicle_plate.py -q
```

Expected: `E   ImportError: cannot import name 'read_vehicle_plate' from 'tvd.plates'`

- [ ] **Step 3: Write the code**

`src/tvd/plates/__init__.py`, replaced as a whole:

```python
"""Plate reading: models behind an interface, the reading of one frame, and the vote over many frames.

Importing this package registers every implementation of the models in it.
"""

from tvd.plates import onnx  # noqa: F401  Registers "onnx_worker".
from tvd.plates.base import PAD, PLATE_MODELS, BasePlateModels, PlateBox, PlateText
from tvd.plates.reading import PlateRead, read_vehicle, select_frames
from tvd.plates.settings import PlatesConfig, ReadingSettings, load_plates_config
from tvd.plates.vehicle import read_vehicle_plate, save_plate_image
from tvd.plates.voting import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateResult, vote

__all__ = [
    "NO_PLATE", "PAD", "PLATE_MODELS", "READ", "UNCERTAIN", "UNREADABLE", "BasePlateModels", "PlateBox", "PlateRead", "PlateResult", "PlateText",
    "PlatesConfig", "ReadingSettings", "load_plates_config", "read_vehicle", "read_vehicle_plate", "save_plate_image", "select_frames", "vote",
]
```

`src/tvd/plates/vehicle.py`:

```python
"""One vehicle's plate, from the frames of a video where it was tracked: a candidate's clip, in the service."""

from __future__ import annotations

from pathlib import Path

import cv2

from tvd.domain import Box
from tvd.plates.base import BasePlateModels
from tvd.plates.reading import read_vehicle, select_frames
from tvd.plates.settings import ReadingSettings
from tvd.plates.voting import PlateResult, vote
from tvd.video import VideoReader


def read_vehicle_plate(video: VideoReader, boxes: dict[int, Box], models: BasePlateModels, settings: ReadingSettings) -> PlateResult:
    """Read the plate in the frames where the vehicle looks largest, and let them vote. ``boxes`` are the vehicle's, by frame of the video."""
    wanted = set(select_frames(boxes, settings.frames_per_track, settings.min_frame_gap))
    reads = []
    if wanted:
        for frame in video.frames(min(wanted), max(wanted) + 1):
            if frame.index in wanted:
                read = read_vehicle(frame.image, frame.index, boxes[frame.index], models, settings)
                if read is not None:
                    reads.append(read)
    return vote(reads, settings)


def save_plate_image(result: PlateResult, folder: str | Path, name: str) -> Path | None:
    """Save the image of the result's best read, without further compression, as ``<name>_f<frame>.png``. None if there is no read."""
    read = result.best_read()
    if read is None:
        return None
    path = Path(folder) / f"{name}_f{read.frame:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), read.image):
        raise RuntimeError(f"Could not write {path}")
    return path
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `223 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_vehicle_plate.py src/tvd/plates/__init__.py src/tvd/plates/vehicle.py
git commit -m "Read a vehicle's plate from the frames of its clip"
```

### Task 8: Evidence images

**Why:** Three images per candidate, before, at and after the trigger, 0.7 s apart, with the vehicle's box. A vehicle hidden when it entered cannot be seen at that moment, so its images are the estimated entry, where it is first seen, and after that (Notebook 07). A rule may draw what it saw on these images through a `FrameOverlay`: an **interface owned by neither side**, in `drawing.py`, so that evidence does not depend on the red-light rule, nor the rule on evidence. The clip itself is never drawn on.

**Files:**
- Change: `tests/tvd/notebook_data.py`
- Create: `tests/tvd/test_tvd_images.py`
- Create: `src/tvd/drawing.py`
- Create: `src/tvd/evidence/images.py`
- Change: `src/tvd/evidence/__init__.py`

**Interfaces:**
- Consumes: `Clip` (Task 3); `ImageSettings` (Task 3); `VideoReader` (Part 1).
- Produces: `FrameOverlay.draw(image, frame_index)`, where `frame_index` is the frame of the source video; `draw_vehicle_box(image, box)`; `EvidenceImage(role, frame_in_clip, source_frame, path)`; `evidence_frames(clip, offset_frames) -> list[tuple[str, int]]`; `write_evidence_images(clip, folder, name, settings, overlay=None) -> list[EvidenceImage]`.

- [ ] **Step 1: Write the failing test**

Without an overlay, the images must equal Notebook 08's, byte for byte.

Add to the end of `tests/tvd/notebook_data.py`:

```python
NOTEBOOK_08_EVIDENCE = NOTEBOOK_08 / "events" / "evidence"


def clip_from_entry(entry: dict):
    """A Notebook 08 clip, as tvd describes a clip."""
    from tvd.evidence import Clip

    start = entry["source_start_frame"]
    boxes = {b["frame_in_clip"]: Box(*b["box_xyxy"]) for b in entry["vehicle_boxes"]}
    return Clip(NOTEBOOK_08_CLIPS / entry["clip_file"], start, entry["source_end_frame"], entry["fps"], entry["width"], entry["height"], start + entry["trigger_frame_in_clip"], boxes)


def notebook_08_records() -> dict[str, dict]:
    """Notebook 08's records, by event id. They hold real plate numbers: never print them."""
    return {r["event_id"]: r for r in load_json(NOTEBOOK_08 / "events" / "traffic_video_modified_review_events.json")["events"]}
```

`tests/tvd/test_tvd_images.py`:

```python
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
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_images.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.drawing'`

- [ ] **Step 3: Write the code**

`src/tvd/drawing.py`:

```python
"""Drawing for a reviewer, on evidence images only: a clip keeps its original pixels."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from tvd.domain import Box

VEHICLE_COLOR = (0, 255, 255)  # BGR: yellow


class FrameOverlay(ABC):
    """What a rule saw at one frame of the source video, drawn on an image of that frame, such as the stop line and the signal state."""

    @abstractmethod
    def draw(self, image: np.ndarray, frame_index: int) -> None:
        """Draw on ``image`` in place. ``frame_index`` is the frame of the source video, not of a clip."""


def draw_vehicle_box(image: np.ndarray, box: Box) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box.as_list())
    cv2.rectangle(image, (x1, y1), (x2, y2), VEHICLE_COLOR, max(2, image.shape[0] // 360))
```

`src/tvd/evidence/images.py`:

```python
"""Three images of each candidate, from its clip: before, at and after the moment the rule fired.

A vehicle hidden at its entry cannot be seen in those frames, and another vehicle may fill them: its
images are then the estimated entry, the moment it is first seen, and after that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from tvd.drawing import FrameOverlay, draw_vehicle_box
from tvd.evidence.clips import Clip
from tvd.evidence.settings import ImageSettings
from tvd.video import VideoReader


@dataclass(frozen=True)
class EvidenceImage:
    role: str
    frame_in_clip: int
    source_frame: int
    path: Path


def evidence_frames(clip: Clip, offset_frames: int) -> list[tuple[str, int]]:
    """The role and the frame of the clip of each image, in order."""
    last = clip.frame_count - 1
    entry = min(clip.trigger_in_clip, last)
    if clip.boxes and entry < min(clip.boxes):
        first_seen = min(clip.boxes)
        return [("estimated_entry", entry), ("first_seen", first_seen), ("after", min(last, first_seen + offset_frames))]
    return [("before", max(0, entry - offset_frames)), ("entry", entry), ("after", min(last, entry + offset_frames))]


def write_evidence_images(clip: Clip, folder: str | Path, name: str, settings: ImageSettings, overlay: FrameOverlay | None = None) -> list[EvidenceImage]:
    """Save the images as ``<name>_<role>.jpg``, with the vehicle's box where it was tracked, over what ``overlay`` draws.

    Two roles that fall on one frame, at the end of a clip, give one image: the first role's.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    roles = evidence_frames(clip, round(settings.offset_seconds * clip.fps))
    wanted: dict[int, str] = {}
    for role, frame in roles:
        wanted.setdefault(frame, role)
    saved: dict[str, EvidenceImage] = {}
    with VideoReader(clip.path) as video:
        for frame in video.frames(min(wanted), max(wanted) + 1):
            if frame.index not in wanted:
                continue
            image, role = frame.image, wanted[frame.index]
            if overlay is not None:
                overlay.draw(image, clip.start + frame.index)
            box = clip.box_near(frame.index, settings.box_tolerance_frames)
            if box is not None:
                draw_vehicle_box(image, box)
            path = folder / f"{name}_{role}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]):
                raise RuntimeError(f"Could not write {path}")
            saved[role] = EvidenceImage(role, frame.index, clip.start + frame.index, path)
    return [saved[role] for role, _ in roles if role in saved]
```

`src/tvd/evidence/__init__.py`, replaced as a whole:

```python
"""The evidence of each candidate: its clip, and images from it."""

from tvd.evidence.clips import Clip, clip_range, cut_clips
from tvd.evidence.images import EvidenceImage, evidence_frames, write_evidence_images
from tvd.evidence.settings import ClipSettings, EvidenceSettings, ImageSettings, RecordSettings, load_evidence_settings

__all__ = [
    "Clip", "ClipSettings", "EvidenceImage", "EvidenceSettings", "ImageSettings", "RecordSettings",
    "clip_range", "cut_clips", "evidence_frames", "load_evidence_settings", "write_evidence_images",
]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `229 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/notebook_data.py tests/tvd/test_tvd_images.py src/tvd/drawing.py src/tvd/evidence/images.py src/tvd/evidence/__init__.py
git commit -m "Save three evidence images of each candidate"
```

### Task 9: The stop line and the signal state on red-light images

**Why:** You asked for red-light images to show the stop line and the signal state. The rule returns a `RedLightOverlay` in its result: at each frame it draws the stop line where it was in that frame (with a moving camera, the line of the camera file is mapped from the reference frame), and a label with the state the rule read for each signal group, after smoothing, which is what it judged. A simulated signal says so on the image. Magenta is used for the line because no signal shows that colour.

**Files:**
- Change: `tests/tvd/test_tvd_red_light.py`
- Change: `tests/tvd/test_tvd_rules.py`
- Change: `src/tvd/drawing.py`
- Change: `src/tvd/rules/base.py`
- Change: `src/tvd/rules/red_light.py`

**Interfaces:**
- Consumes: `FrameOverlay` (Task 8); `CameraMotion.to_frame` and `SignalTimeline` (Part 2); `RuleResult` (Part 2).
- Produces: `RuleResult.overlay: FrameOverlay | None`, None by default; `RedLightOverlay(stop_line, motion, signals, groups, signal_source)` with `lines(frame_index) -> list[tuple[str, colour]]` and `draw(image, frame_index)`; `draw_label(image, lines)`. A group without heads is not shown.

- [ ] **Step 1: Write the failing test**

In `tests/tvd/test_tvd_red_light.py`:

Replace:

```python
from tvd.rules.red_light import RedLightRule
```

with:

```python
from tvd.rules.red_light import RedLightOverlay, RedLightRule
```

Add to the end of the file:

```python
MAGENTA = [255, 0, 255]


def test_the_overlay_draws_the_stop_line_and_the_signal_state():
    overlay = judge(car_up()).overlay
    image = np.zeros((360, 640, 3), np.uint8)
    overlay.draw(image, 31)
    assert overlay.lines(31) == [("through signal: red", (0, 0, 255))]  # The left group has no heads here, so it is not shown.
    assert image[200, 320].tolist() == MAGENTA and (image[:40, :300] == [0, 0, 255]).all(axis=2).any()


def test_the_overlay_follows_a_moving_camera():
    motion = CameraMotion(np.tile([1.0, 0.0, 0.0, 10.0], (60, 1)))  # Each frame is 10 px above the reference frame.
    signals = SignalTimeline({"through": {5: "green"}})
    overlay = RedLightOverlay([(0, 200), (640, 200)], motion, signals, ["through"], "camera")
    image = np.zeros((360, 640, 3), np.uint8)
    overlay.draw(image, 5)
    assert image[190, 320].tolist() == MAGENTA and image[200, 320].tolist() != MAGENTA and overlay.lines(5) == [("through signal: green", (0, 200, 0))]


def test_a_simulated_signal_says_so_on_the_image():
    overlay = RedLightOverlay([(0, 200), (640, 200)], CameraMotion(), SignalTimeline({"through": {}}), ["through"], "simulated_overlay")
    assert overlay.lines(0) == [("through signal: unknown", (170, 170, 170)), ("signal source: simulated_overlay", (255, 255, 255))]
```

Add to the end of `tests/tvd/test_tvd_rules.py`:

```python
def test_a_rule_has_no_overlay_unless_it_draws_one():
    assert RuleResult([], [], {}).overlay is None
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_red_light.py tests/tvd/test_tvd_rules.py -q
```

Expected: `1 error during collection`: `E   ImportError: cannot import name 'RedLightOverlay' from 'tvd.rules.red_light'`. pytest stops there, so the test of `test_tvd_rules.py` does not run yet; alone, it would fail with `AttributeError: 'RuleResult' object has no attribute 'overlay'`.

- [ ] **Step 3: Write the code**

Add to the end of `src/tvd/drawing.py`:

```python
def draw_label(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Lines of text, each in its own colour, on a dark band at the top left of the image."""
    scale = max(0.5, 0.8 * image.shape[0] / 1080)
    thickness = max(1, round(scale * 2.5))
    sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for text, _ in lines]
    step = max(height for _, height in sizes) + 12
    cv2.rectangle(image, (0, 0), (max(width for width, _ in sizes) + 16, step * len(lines) + 4), (0, 0, 0), -1)
    for k, ((text, color), (_, height)) in enumerate(zip(lines, sizes)):
        cv2.putText(image, text, (8, step * k + height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
```

In `src/tvd/rules/base.py`:

Before:

```python
from tvd.motion import CameraMotion
```

add:

```python
from tvd.drawing import FrameOverlay
```

Replace:

```python
    """A rule's findings, a record of how it judged each track, and the settings it used."""
```

with:

```python
    """A rule's findings, a record of how it judged each track, and the settings it used.

    ``overlay``, if the rule has one, draws on an evidence image what the rule saw at that frame.
    """
```

After:

```python
    records: list[dict[str, Any]]
    settings: dict[str, Any]
```

add:

```python
    overlay: FrameOverlay | None = None
```

In `src/tvd/rules/red_light.py`:

Before:

```python
import numpy as np
from pydantic import Field, field_validator
```

add:

```python
import cv2
```

After:

```python
from tvd.config import StrictModel, validate
```

add:

```python
from tvd.drawing import FrameOverlay, draw_label
```

Replace:

```python
from tvd.signals import SIGNAL_READERS, UNKNOWN, SignalMonitor, SignalTimeline
```

with:

```python
from tvd.signals import GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, SignalMonitor, SignalTimeline
```

Before:

```python
@RULES.register("red_light")
class RedLightRule(BaseRule):
```

add:

```python
STATE_COLORS = {RED: (0, 0, 255), YELLOW: (0, 200, 255), GREEN: (0, 200, 0)}  # BGR; any other state is gray.
STOP_LINE_COLOR = (255, 0, 255)  # Magenta: no signal shows this colour.


class RedLightOverlay(FrameOverlay):
    """The stop line where it was in each frame, and the state the rule read for each signal group then."""

    def __init__(self, stop_line: list[tuple[float, float]], motion: CameraMotion, signals: SignalTimeline, groups: list[str], signal_source: str) -> None:
        self.stop_line = stop_line
        self.motion = motion
        self.signals = signals
        self.groups = groups
        self.signal_source = signal_source

    def lines(self, frame_index: int) -> list[tuple[str, tuple[int, int, int]]]:
        lines = []
        for group in self.groups:
            state = self.signals.state(group, frame_index)
            lines.append((f"{group} signal: {state}", STATE_COLORS.get(state, (170, 170, 170))))
        if self.signal_source != "camera":
            lines.append((f"signal source: {self.signal_source}", (255, 255, 255)))
        return lines

    def draw(self, image: np.ndarray, frame_index: int) -> None:
        line = self.motion.to_frame(frame_index, self.stop_line)  # The camera file draws the line on the reference frame.
        cv2.polylines(image, [np.round(line).astype(np.int32)], False, STOP_LINE_COLOR, max(2, image.shape[0] // 360), cv2.LINE_AA)
        draw_label(image, self.lines(frame_index))
```

Replace:

```python
        return RuleResult(findings, records, s.model_dump())
```

with:

```python
        groups = [group for group in SIGNAL_GROUPS if getattr(self.scene.signals, group)]  # A group without heads is never drawn.
        overlay = RedLightOverlay(self.scene.stop_line, motion, signals, groups, self.scene.signal_source)
        return RuleResult(findings, records, s.model_dump(), overlay)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `233 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_red_light.py tests/tvd/test_tvd_rules.py src/tvd/drawing.py src/tvd/rules/base.py src/tvd/rules/red_light.py
git commit -m "Draw the stop line and the signal state on red-light evidence"
```

### Task 10: Records, schema version 2

**Why:** A record joins a finding, its evidence and its plate, for the backend. Version 2 adds what Section 6 of the spec asks for: the camera, `recorded_at` and absolute times, the effective settings, and the version of each settings file. Everything else keeps the meaning of version 1, and the tests prove it: the fields kept from version 1 equal those of Notebook 08's two red-light records and Notebook 07's ten wrong-way records. A record is always `pending`: validation refuses anything that looks decided, and any path outside the folder of the records.

**Files:**
- Change: `tests/tvd/notebook_data.py`
- Create: `tests/tvd/test_tvd_records.py`
- Create: `src/tvd/records.py`
- Change: `docs/event_schema.md`

**Interfaces:**
- Consumes: `Finding` (Tasks 2, Part 2); `Clip` (Task 3); `EvidenceImage` (Task 8); `PlateResult` and its statuses (Task 5).
- Produces: `SCHEMA_VERSION = 2`; `parse_recorded_at(text) -> datetime`, which refuses a time without a time zone; `time_at(recorded_at, seconds) -> str`; `RecordContext(video_id, fps, width, height, camera_id, recorded_at, tracking_session, settings, config_files)`; `event_id(video_id, finding)`; `build_record(context, finding, clip, images, plate, plate_image, folder) -> dict`; `validate_record(record) -> list[str]`; `deduplicate_records(records, window_seconds) -> (kept, dropped)`; `save_records(records, path)`.

- [ ] **Step 1: Write the failing test**

A record with `plate=None` says that plate reading was not attempted: that is how a failed plate step keeps its violations (Task 11).

Add to the end of `tests/tvd/notebook_data.py`:

```python
NOTEBOOK_07_REVERSED = OUTPUTS / "07_violation_event_generation" / "wrong_way_control" / "video-test-reversed_review_events.json"
REVERSED_CANDIDATES = OUTPUTS / "05_violation_detection" / "time_reversed_control" / "video-test-reversed_events.json"


def finding_from_event(event: dict, points: tuple = ()):
    """A finding saying what a notebook's candidate event says. A red-light event has no start or end: its points give them."""
    from tvd.rules import Finding

    evidence = {**({"region_id": event["region_id"]} if "region_id" in event else {}), **event["evidence"]}
    if event.get("signal_source"):
        evidence["signal_source"] = event["signal_source"]
    start = event.get("start_frame", points[0].frame_index if points else event["trigger_frame"])
    end = event.get("end_frame", points[-1].frame_index if points else event["trigger_frame"])
    return Finding(event["violation_type"], event["track_id"], event["class_name"], start, event["trigger_frame"], end, event["why"], evidence, points)
```

`tests/tvd/test_tvd_records.py`:

```python
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
RECORDED_AT = parse_recorded_at("2026-11-07T08:00:00+07:00")
FOLDER = Path("/jobs/42")
needs_notebook_outputs = pytest.mark.skipif(
    not (NOTEBOOK_08.is_dir() and NOTEBOOK_07_REVERSED.is_file() and REVERSED_CANDIDATES.is_file()), reason="Needs the outputs of Notebooks 05, 07 and 08, as on the GPU server.",
)


def context(video_id: str = "cam1-0800", fps: float = 25.0) -> RecordContext:
    return RecordContext(video_id, fps, 1920, 1080, "cam1", RECORDED_AT, f"{video_id}/ocsort", SETTINGS, {"camera": "sha256:0123456789ab"})


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
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_records.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.records'`

- [ ] **Step 3: Write the code**

`src/tvd/records.py`:

```python
"""Records of candidates, schema version 2 (``docs/event_schema.md``): one per candidate, for a person to review.

A record joins what the pass found (the finding), its evidence (the clip and its images), and its plate.
It is always ``pending``: nothing in the service confirms a violation. Reviewing is the backend's job.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tvd.evidence import Clip, EvidenceImage
from tvd.plates import NO_PLATE, READ, UNCERTAIN, UNREADABLE, PlateResult
from tvd.rules import Finding

SCHEMA_VERSION = 2
PENDING = "pending"
NOT_ATTEMPTED = "not_attempted"
PLATE_STATUSES = (READ, UNCERTAIN, UNREADABLE, NO_PLATE, NOT_ATTEMPTED)
PLATE_ATTENTION = {UNCERTAIN: "plate_uncertain", UNREADABLE: "plate_unreadable", NO_PLATE: "plate_not_found", NOT_ATTEMPTED: "plate_not_attempted"}
REQUIRED_FIELDS = {
    "schema_version": int, "event_id": str, "camera_id": str, "video": dict, "violation_type": str, "track": dict, "time": dict, "rule": dict,
    "why": str, "signal_source": (str, type(None)), "measurements": dict, "evidence": dict, "license_plate": dict, "needs_attention": list,
    "settings": dict, "config_files": dict, "review": dict,
}


def parse_recorded_at(text: str) -> datetime:
    """The time of a video's first frame, in ISO 8601 with its time zone, such as ``2026-11-07T08:00:00+07:00``."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"recorded_at is not an ISO 8601 time: {text!r}") from error
    if moment.tzinfo is None:
        raise ValueError(f"recorded_at needs a time zone, such as 2026-11-07T08:00:00+07:00: {text!r}")
    return moment


def time_at(recorded_at: datetime, seconds: float) -> str:
    """The absolute time of a moment of the video, rounded to the millisecond as ``*_time_s`` are, in the time zone that was sent."""
    return (recorded_at + timedelta(milliseconds=round(seconds * 1000))).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RecordContext:
    """What every record of one video shares.

    ``settings`` are the effective settings, as ``{"perception": ..., "rules": {name: ...}, "plates": ..., "evidence": ...}``,
    and ``config_files`` the version of each settings file that gave them.
    """

    video_id: str
    fps: float
    width: int
    height: int
    camera_id: str
    recorded_at: datetime
    tracking_session: str
    settings: dict[str, Any]
    config_files: dict[str, str]


def event_id(video_id: str, finding: Finding) -> str:
    """Unique within a video: one rule fires once per track and frame. It never equals a track id."""
    return f"evt-{video_id}-{finding.rule}-f{finding.trigger_frame:05d}-t{finding.track_id}"


def plate_summary(result: PlateResult | None) -> dict[str, Any]:
    """What a record keeps of a plate result. Without one, plate reading was not attempted."""
    if result is None:
        return {"status": NOT_ATTEMPTED, "text": None, "votes": 0, "usable_reads": 0, "agreement": 0.0, "mean_prob": 0.0}
    return {"status": result.status, "text": result.text, "votes": result.votes, "usable_reads": result.usable_reads, "agreement": result.agreement, "mean_prob": result.mean_prob}


def attention_flags(finding: Finding, plate: dict[str, Any]) -> list[str]:
    """What a reviewer should look at closely, because the record alone cannot settle it."""
    flags = []
    if finding.evidence.get("signal_source") == "simulated_overlay":
        flags.append("simulated_signal")
    if finding.evidence.get("entry_estimated"):
        flags.append("entry_estimated")
    elif finding.evidence.get("entry_seen") is False:
        flags.append("entry_not_seen")
    if plate["status"] in PLATE_ATTENTION:
        flags.append(PLATE_ATTENTION[plate["status"]])
    return flags


def relative_to(path: Path, folder: Path) -> str:
    return Path(os.path.relpath(path, folder)).as_posix()


def build_record(
    context: RecordContext, finding: Finding, clip: Clip, images: list[EvidenceImage],
    plate: PlateResult | None, plate_image: Path | None, folder: Path,
) -> dict[str, Any]:
    """The record of one finding. Evidence paths are relative to ``folder``, where the records are saved."""
    fps, recorded_at = context.fps, context.recorded_at
    plate_fields = plate_summary(plate)
    frames = {"start": finding.start_frame, "trigger": finding.trigger_frame, "end": finding.end_frame}
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id(context.video_id, finding),
        "camera_id": context.camera_id,
        "video": {"video_id": context.video_id, "fps": fps, "width": context.width, "height": context.height, "recorded_at": recorded_at.isoformat()},
        "violation_type": finding.rule,
        "track": {"tracking_session": context.tracking_session, "track_id": finding.track_id, "class_name": finding.class_name},
        "time": {
            **{f"{name}_frame": frame for name, frame in frames.items()},
            **{f"{name}_time_s": round(frame / fps, 3) for name, frame in frames.items()},
            **{f"{name}_at": time_at(recorded_at, frame / fps) for name, frame in frames.items()},
        },
        "rule": {"name": finding.rule, "version": context.settings["rules"][finding.rule]["version"]},
        "why": finding.why,
        "signal_source": finding.evidence.get("signal_source"),
        "measurements": {k: v for k, v in finding.evidence.items() if k != "signal_source"},
        "evidence": {
            "clip": relative_to(clip.path, folder),
            "clip_frames": [clip.start, clip.end],
            "plate_crop": relative_to(plate_image, folder) if plate_image else None,
            "frames": [{"role": i.role, "frame_in_clip": i.frame_in_clip, "source_frame": i.source_frame, "image": relative_to(i.path, folder)} for i in images],
        },
        "license_plate": plate_fields,
        "needs_attention": attention_flags(finding, plate_fields),
        "settings": {
            "perception": context.settings["perception"], "rule": context.settings["rules"][finding.rule],
            "plates": context.settings["plates"], "evidence": context.settings["evidence"],
        },
        "config_files": context.config_files,
        "review": {"status": PENDING, "reviewer": None, "decided_at": None, "corrected_plate": None, "notes": None},
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Everything wrong with a record, as sentences. An empty list means it is fit for review."""
    problems = [f"{name} is missing" for name in REQUIRED_FIELDS if name not in record]
    problems += [f"{name} has the wrong type" for name, kind in REQUIRED_FIELDS.items() if name in record and not isinstance(record[name], kind)]
    if problems:
        return problems
    video_id, times, plate, evidence = record["video"].get("video_id"), record["time"], record["license_plate"], record["evidence"]
    if record["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if not video_id or video_id not in record["event_id"] or record["event_id"] == str(record["track"].get("track_id")):
        problems.append("event_id must name its video, and differ from the track id")
    if not str(record["track"].get("tracking_session", "")).startswith(f"{video_id}/"):
        problems.append("the track must be scoped to a tracking session of this video")
    if not (isinstance(times.get("trigger_frame"), int) and times["trigger_frame"] >= 0 and times.get("trigger_time_s", -1) >= 0):
        problems.append("the trigger time is invalid")
    try:
        parse_recorded_at(times.get("trigger_at", ""))
    except (TypeError, ValueError):
        problems.append("the trigger's absolute time needs a time zone")
    if plate.get("status") not in PLATE_STATUSES:
        problems.append("the plate status is unknown")
    elif (plate.get("text") is None) != (plate["status"] not in (READ, UNCERTAIN)):
        problems.append("the plate text does not fit its status")
    if record["review"].get("status") != PENDING or record["review"].get("reviewer") is not None:
        problems.append("a record from the service is pending, with no reviewer")
    paths = [evidence.get("clip"), evidence.get("plate_crop"), *(f.get("image") for f in evidence.get("frames", []))]
    if any(p is not None and (Path(p).is_absolute() or ".." in Path(p).parts) for p in paths):
        problems.append("evidence paths must be relative, inside the folder of the records")
    return problems


def same_occurrence(a: dict[str, Any], b: dict[str, Any], window_seconds: float) -> bool:
    """True if two records describe one violation: the same id, or one vehicle shortly apart."""
    if a["event_id"] == b["event_id"]:
        return True
    if a["video"]["video_id"] != b["video"]["video_id"] or a["violation_type"] != b["violation_type"]:
        return False
    if abs(a["time"]["trigger_time_s"] - b["time"]["trigger_time_s"]) > window_seconds:
        return False
    plates = a["license_plate"], b["license_plate"]
    same_plate = all(p["status"] == READ for p in plates) and plates[0]["text"] == plates[1]["text"]  # An uncertain text may be wrong.
    return a["track"]["track_id"] == b["track"]["track_id"] or same_plate


def deduplicate_records(records: list[dict[str, Any]], window_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first record of each occurrence. Returns the records kept, and those dropped with what they repeat."""
    kept: list[dict[str, Any]] = []
    dropped = []
    for record in sorted(records, key=lambda r: r["time"]["trigger_time_s"]):
        original = next((k for k in kept if same_occurrence(k, record, window_seconds)), None)
        if original is None:
            kept.append(record)
        else:
            dropped.append({**record, "duplicate_of": original["event_id"]})
    return kept, dropped


def save_records(records: list[dict[str, Any]], path: Path) -> None:
    """Save records as JSON. They hold real plate numbers, so the file must stay out of git."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "events": records}, indent=1))
```

In `docs/event_schema.md`:

Before:

```markdown
## Candidate record

| Field | Type | Meaning |
```

add:

```markdown
## Version 2: the review service

The review service (`src/tvd/records.py`) writes version 2. It keeps every field of version 1 below, with the same meaning, and changes these:

| Field | Version 2 |
| --- | --- |
| `schema_version` | 2 |
| `event_id` | `evt-<video_id>-<rule>-f<trigger frame>-t<track id>`, for both rules |
| `camera_id` | New: the camera file the video was checked with |
| `video` | Adds `recorded_at`: the time of the video's first frame, as the backend sent it, in ISO 8601 with its time zone |
| `track` | `tracking_session` is `<video_id>/<tracker>`. There is no `clip`: a camera file describes the whole video |
| `time` | Adds `start_at`, `trigger_at` and `end_at`: `recorded_at` plus the seconds into the video, to the millisecond, in the same time zone. Red-light records now have a start and an end too: the first and last frames of the vehicle's track |
| `rule` | Only `name` and `version`. All the rule's settings are in `settings.rule` |
| `evidence` | Paths are relative to the folder of `records.json`: `clips/`, `evidence/` and `plates/`. Adds `clip_frames`: the clip's first frame and one past its last, in the video |
| `settings` | New: the effective settings that produced the record, as `perception`, `rule`, `plates` and `evidence` |
| `config_files` | New: the version of each settings file, as `sha256:` and the first 12 hex digits of its hash |
| `review` | Always `pending`, with no reviewer: the service never decides. Review happens in the backend |

The file is `{"schema_version": 2, "events": [...]}`.
```

After:

```markdown
The file is `{"schema_version": 2, "events": [...]}`.

## Candidate record
```

add:

```markdown

```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `248 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/notebook_data.py tests/tvd/test_tvd_records.py src/tvd/records.py docs/event_schema.md
git commit -m "Build version 2 records of the candidates"
```

### Task 11: Evidence and records for every candidate

**Why:** `review_check` runs Tasks 3 to 10 on the result of Part 2's pass: clips, plates, images, records, then duplicates and validation, and writes one folder. It takes components that are already built (**dependency inversion**), so Part 4's job queue can call it with plate models loaded once. **A failed plate reading never loses a violation:** each candidate's reading is tried on its own, and a failure is noted in `summary.json` while the record says `not_attempted`. An output folder that is not new or empty is refused, because files of an earlier run could show the wrong moment.

**Files:**
- Change: `tests/tvd/test_tvd_check.py`
- Change: `tests/tvd/test_tvd_config.py`
- Create: `tests/tvd/test_tvd_review.py`
- Change: `src/tvd/check.py`
- Change: `src/tvd/config.py`
- Create: `src/tvd/review.py`

**Interfaces:**
- Consumes: `CheckResult` (Part 2); `cut_clips`, `write_evidence_images`, `EvidenceSettings` (Tasks 3, 8); `read_vehicle_plate`, `save_plate_image` (Task 7); `RecordContext`, `build_record`, `deduplicate_records`, `validate_record`, `save_records`, `event_id` (Task 10); `RuleResult.overlay` (Task 9).
- Produces: `CheckResult.first_frame` (the analysed frames are `first_frame` to `first_frame + frames`); `file_version(path) -> str` in `config.py`; `prepare_folder(folder)`; `review_check(reader, check, context, plate_models, reading, evidence, folder, plates_unavailable="no plate models") -> ReviewResult`; `ReviewResult(folder, check, records, duplicates, failures, seconds)` with `summary()`, which holds counts only.

- [ ] **Step 1: Write the failing test**

In `tests/tvd/test_tvd_check.py`:

Replace:

```python
def run(sample_video, camera: dict = CAMERA):
```

with:

```python
def run(sample_video, camera: dict = CAMERA, start: int = 0):
```

Replace:

```python
            build_tracker(ComponentConfig(type="test-same-id")), {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS),
```

with:

```python
            build_tracker(ComponentConfig(type="test-same-id")), {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS), start,
```

Add to the end of the file:

```python
def test_the_first_frame_analysed_is_kept(sample_video):
    result = run(sample_video, start=100)
    assert (result.first_frame, result.frames, result.as_json()["first_frame"]) == (100, 140, 100)
```

In `tests/tvd/test_tvd_config.py`:

Replace:

```python
from tvd.config import ConfigError, PerceptionConfig, load_config, validate
```

with:

```python
from tvd.config import ConfigError, PerceptionConfig, file_version, load_config, validate
```

Add to the end of the file:

```python
def test_a_file_version_changes_with_its_content(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("window_seconds: 2.0\n")
    first = file_version(path)
    path.write_text("window_seconds: 2.5\n")
    assert first.startswith("sha256:") and len(first) == len("sha256:") + 12 and file_version(path) != first
```

`tests/tvd/test_tvd_review.py`:

```python
import json

import cv2
import numpy as np
import pytest
from conftest import ROOT, FakePlateModels

from tvd.camera import CameraConfig, load_rule_defaults
from tvd.check import check_video
from tvd.config import ComponentConfig
from tvd.evidence import load_evidence_settings
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
from tvd.plates import load_plates_config
from tvd.records import RecordContext, parse_recorded_at, validate_record
from tvd.review import review_check
from tvd.video import VideoReader

DEFAULTS = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
PLATES = load_plates_config(ROOT / "configs" / "tvd" / "plates.yaml")
EVIDENCE = load_evidence_settings(ROOT / "configs" / "tvd" / "evidence.yaml")
# As in test_tvd_check.py: the test car comes down the image by 1 px per frame, through a red light, against the only allowed direction.
CAMERA = {
    "camera_id": "test-camera",
    "frame_size": [640, 360],
    "rules": {
        "red_light": {
            "scene": {"stop_line": [[0, 150], [640, 150]], "lane_dividers": [0, 600], "lane_direction_deg": 90, "traffic_direction": "toward_camera",
                      "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}},
            "settings": {"entry_offset": 0.0, "pass_margin": 0.2},
        },
        "wrong_way": {"scene": {"vanishing_point": [320, -100000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360], [0, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}},
    },
}


class BrokenPlateModels(FakePlateModels):
    def detect(self, image):
        raise RuntimeError("The plate models' process stopped")


def review(video, folder, plate_models=None, camera: dict = CAMERA):
    config = CameraConfig.model_validate(camera)
    with VideoReader(video) as reader:
        check = check_video(
            reader, config, build_detector(ComponentConfig(type="test-moving-car", options={"step": 0, "step_y": 1})),
            build_tracker(ComponentConfig(type="test-same-id")), {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS),
        )
        settings = {"perception": {"tracker": "test-same-id"}, "rules": {n: r.settings for n, r in check.results.items()}, "plates": PLATES.model_dump(), "evidence": EVIDENCE.model_dump()}
        context = RecordContext("traffic_sample", reader.fps, reader.width, reader.height, "test-camera", parse_recorded_at("2026-11-07T08:00:00+07:00"),
                                "traffic_sample/test-same-id", settings, {"camera": "sha256:000000000000"})
        return review_check(reader, check, context, plate_models, PLATES.reading, EVIDENCE, folder)


def test_a_folder_with_the_evidence_and_records_of_every_candidate(tmp_path, sample_video):
    result = review(sample_video, tmp_path / "job", FakePlateModels())
    folder = tmp_path / "job"
    saved = json.loads((folder / "records.json").read_text())
    assert saved["schema_version"] == 2 and [r["event_id"] for r in saved["events"]] == ["evt-traffic_sample-wrong_way-f00048-t1", "evt-traffic_sample-red_light-f00070-t1"]
    assert sorted(p.name for p in (folder / "clips").iterdir()) == ["evt-traffic_sample-red_light-f00070-t1.mp4", "evt-traffic_sample-wrong_way-f00048-t1.mp4"]
    assert len(list((folder / "evidence").glob("*.jpg"))) == 6 and len(list((folder / "plates").glob("*.png"))) == 2
    for record in result.records:
        paths = [record["evidence"]["clip"], record["evidence"]["plate_crop"], *(f["image"] for f in record["evidence"]["frames"])]
        assert validate_record(record) == [] and all((folder / p).is_file() for p in paths) and record["review"]["status"] == "pending"
    assert json.loads((folder / "summary.json").read_text())["plates"] == {"unreadable": 2}  # The test car's plate is narrower than 30 px.


def test_red_light_images_show_the_stop_line(tmp_path, sample_video):
    result = review(sample_video, tmp_path, FakePlateModels())
    red = next(r for r in result.records if r["violation_type"] == "red_light")
    entry = cv2.imread(str(tmp_path / red["evidence"]["frames"][1]["image"])).astype(int)
    magenta = (entry[148:153, :, 0] > 200) & (entry[148:153, :, 1] < 80) & (entry[148:153, :, 2] > 200)
    assert magenta.any(axis=0).mean() > 0.8  # The line at y = 150 crosses the whole image, behind the label and the car.


def test_a_failed_plate_reading_keeps_every_record(tmp_path, sample_video):
    result = review(sample_video, tmp_path, BrokenPlateModels())
    assert len(result.records) == 2 and all(r["license_plate"]["status"] == "not_attempted" for r in result.records)
    assert all("plate_not_attempted" in r["needs_attention"] for r in result.records) and len(result.failures) == 2
    assert "The plate models' process stopped" in json.loads((tmp_path / "summary.json").read_text())["failures"][0]


def test_without_plate_models_the_records_say_so(tmp_path, sample_video):
    result = review(sample_video, tmp_path, None)
    assert result.failures == ["plate reading: no plate models"] and {r["license_plate"]["status"] for r in result.records} == {"not_attempted"}


def test_a_folder_already_in_use_is_refused(tmp_path, sample_video):
    (tmp_path / "records.json").write_text("{}")
    with pytest.raises(FileExistsError, match="new or empty folder"):
        review(sample_video, tmp_path, FakePlateModels())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["records.json"]


def test_a_file_where_the_folder_should_be_is_refused(tmp_path, sample_video):
    (tmp_path / "job").write_text("not a folder")
    with pytest.raises(FileExistsError, match="new or empty folder"):
        review(sample_video, tmp_path / "job", FakePlateModels())


def test_no_candidate_gives_an_empty_list_of_records(tmp_path, sample_video):
    calm = {**CAMERA, "rules": {"wrong_way": {"scene": {**CAMERA["rules"]["wrong_way"]["scene"], "regions": [
        {**CAMERA["rules"]["wrong_way"]["scene"]["regions"][0], "allowed_direction": "toward_camera"}]}}}}
    result = review(sample_video, tmp_path, FakePlateModels(), calm)
    assert result.records == [] and json.loads((tmp_path / "records.json").read_text())["events"] == [] and not (tmp_path / "clips").exists()
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_check.py tests/tvd/test_tvd_config.py tests/tvd/test_tvd_review.py -q
```

Expected: `2 errors during collection`: `E   ImportError: cannot import name 'file_version' from 'tvd.config'` and `E   ModuleNotFoundError: No module named 'tvd.review'`. pytest stops there, so the new test of `test_tvd_check.py` does not run yet.

- [ ] **Step 3: Write the code**

In `src/tvd/check.py`:

After:

```python
    seconds: float
    results: dict[str, RuleResult]
```

add:

```python
    first_frame: int = 0  # The analysed frames are first_frame to first_frame + frames.
```

Replace:

```python
            "video": self.video, "camera_id": self.camera_id, "frames": self.frames, "fps": self.fps, "tracks": self.tracks, "seconds": self.seconds,
```

with:

```python
            "video": self.video, "camera_id": self.camera_id, "first_frame": self.first_frame, "frames": self.frames, "fps": self.fps, "tracks": self.tracks,
            "seconds": self.seconds,
```

Replace:

```python
    return CheckResult(reader.path.name, camera.camera_id, history.frames_seen, reader.fps, len(history.tracks), round(time.perf_counter() - began, 1), results)
```

with:

```python
    seconds = round(time.perf_counter() - began, 1)
    return CheckResult(reader.path.name, camera.camera_id, history.frames_seen, reader.fps, len(history.tracks), seconds, results, start)
```

In `src/tvd/config.py`:

Before:

```python
from pathlib import Path
from typing import Any, TypeVar
```

add:

```python
import hashlib
```

Add to the end of the file:

```python
def file_version(path: str | Path) -> str:
    """A settings file's version, from its content: any change to the file changes it. Records keep it."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
```

`src/tvd/review.py`:

```python
"""After the pass (``check.py``): each candidate's clip, plate, images and record, in one folder for review.

The folder holds ``records.json``, ``summary.json``, ``clips/``, ``evidence/`` and ``plates/``. A failed plate
reading never loses what was found: the record is kept, and says that plate reading was not attempted.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tvd.check import CheckResult
from tvd.evidence import EvidenceSettings, cut_clips, write_evidence_images
from tvd.plates import BasePlateModels, PlateResult, ReadingSettings, read_vehicle_plate, save_plate_image
from tvd.records import RecordContext, build_record, deduplicate_records, event_id, save_records, validate_record
from tvd.video import VideoReader


@dataclass
class ReviewResult:
    folder: Path
    check: CheckResult
    records: list[dict[str, Any]]
    duplicates: list[dict[str, Any]]
    failures: list[str]
    seconds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Counts only: a summary never holds a plate number."""
        return {
            "video": self.check.video, "camera_id": self.check.camera_id, "frames": self.check.frames, "tracks": self.check.tracks,
            "verdicts": {name: r.verdict_counts() for name, r in self.check.results.items()},
            "records": len(self.records), "duplicates_dropped": len(self.duplicates),
            "plates": dict(Counter(r["license_plate"]["status"] for r in self.records)),
            "failures": self.failures, "seconds": {"check": self.check.seconds, **self.seconds},
        }


def prepare_folder(folder: Path) -> None:
    """A new folder, or an empty one: files of an earlier run could show the wrong moment."""
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError(f"The output must be a new or empty folder: {folder}")
    folder.mkdir(parents=True, exist_ok=True)


def read_plates(
    clips: dict, models: BasePlateModels | None, settings: ReadingSettings, folder: Path, failures: list[str], unavailable: str,
) -> dict[str, tuple[PlateResult, Path | None]]:
    """Each candidate's plate, from its clip. A candidate whose reading fails is left out, and the failure noted."""
    if models is None:
        failures.append(f"plate reading: {unavailable}")
        return {}
    plates = {}
    for name, clip in clips.items():
        try:
            with VideoReader(clip.path) as video:
                result = read_vehicle_plate(video, clip.boxes, models, settings)
            plates[name] = (result, save_plate_image(result, folder, name))
        except Exception as error:  # Any failure here must not lose the violation: its record says the plate was not read.
            failures.append(f"plate reading of {name}: {type(error).__name__}: {error}")
    return plates


def review_check(
    reader: VideoReader, check: CheckResult, context: RecordContext, plate_models: BasePlateModels | None,
    reading: ReadingSettings, evidence: EvidenceSettings, folder: str | Path, plates_unavailable: str = "no plate models",
) -> ReviewResult:
    """Write the evidence and the records of every finding of a pass over ``reader``'s video.

    Without ``plate_models``, every record says that plate reading was not attempted, and ``plates_unavailable`` says why.
    """
    folder = Path(folder)
    prepare_folder(folder)
    findings = sorted(((name, f) for name, r in check.results.items() for f in r.findings), key=lambda item: item[1].trigger_frame)
    by_id = {event_id(context.video_id, f): (name, f) for name, f in findings}
    seconds, failures = {}, []

    began = time.perf_counter()
    clips = cut_clips(reader, {k: f for k, (_, f) in by_id.items()}, check.first_frame, check.first_frame + check.frames, folder / "clips", evidence.clip)
    seconds["clips"] = round(time.perf_counter() - began, 1)

    began = time.perf_counter()
    plates = read_plates(clips, plate_models, reading, folder / "plates", failures, plates_unavailable)
    seconds["plates"] = round(time.perf_counter() - began, 1)

    began = time.perf_counter()
    records = []
    for key, (rule, finding) in by_id.items():
        images = write_evidence_images(clips[key], folder / "evidence", key, evidence.images, check.results[rule].overlay)
        plate, plate_image = plates.get(key, (None, None))
        records.append(build_record(context, finding, clips[key], images, plate, plate_image, folder))
    kept, dropped = deduplicate_records(records, evidence.records.duplicate_seconds)
    problems = {r["event_id"]: p for r in kept if (p := validate_record(r))}
    if problems:
        raise ValueError(f"Records not fit for review: {problems}")  # A broken record must never reach a reviewer.
    save_records(kept, folder / "records.json")
    seconds["images and records"] = round(time.perf_counter() - began, 1)

    result = ReviewResult(folder, check, kept, dropped, failures, seconds)
    (folder / "summary.json").write_text(json.dumps(result.summary(), indent=1))
    return result
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `257 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_check.py tests/tvd/test_tvd_config.py tests/tvd/test_tvd_review.py src/tvd/check.py src/tvd/config.py src/tvd/review.py
git commit -m "Write the evidence and records of every candidate in one folder"
```

### Task 12: The review command

**Why:** The composition root of this part: it reads every settings file, the time and the output folder before any model loads, starts the plate models once, runs the pass and the review, and prints counts only. The last test is the one Section 9 of the spec asks for: clip C of `rlr_occasions.mp4`, cut into a file of its own with a camera file of its own, must give Notebook 08's records. FFV1 cuts it without loss, so the frames, and so everything after them, can be compared exactly.

**Files:**
- Change: `tests/tvd/notebook_data.py`
- Change: `tests/tvd/test_tvd_cli.py`
- Create: `tests/tvd/test_tvd_privacy.py`
- Change: `src/tvd/cli.py`
- Create: `configs/cameras/rlr_occasions-C.yaml`

**Interfaces:**
- Consumes: Everything above; `run_check`, `check_summary` (Part 2).
- Produces: `python -m tvd.cli review --video --camera --recorded-at --output [--perception] [--rules] [--plates] [--evidence]`; `run_review(...) -> ReviewResult`; `review_summary(result) -> str`. Exit code 2, with one line, for every refused input.

- [ ] **Step 1: Write the failing test**

`CannotStart` is registered here, in the test file, because only these tests need plate models that fail to start. `test_tvd_privacy.py` is a guard, as `tests/test_plates.py` has one for the notebooks: from this task on, real plates exist in `outputs/`, and no file that goes to GitHub may hold one. It reads the private labels, so it skips where they are not; it passes before the code too.

Add to the end of `tests/tvd/notebook_data.py`:

```python
NOTEBOOK_08_RLR = OUTPUTS / "08_end_to_end_baseline" / "rlr_occasions"


def cut_losslessly(source: Path, start: int, end: int, path: Path) -> Path:
    """Frames ``start`` to ``end`` (not included) of a video, in a file of their own that decodes to exactly the same pixels (FFV1)."""
    import subprocess

    from tvd.video import VideoReader, find_ffmpeg

    with VideoReader(source) as reader:
        size, fps = f"{reader.width}x{reader.height}", reader.fps
        command = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", size, "-r", str(fps), "-i", "-", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame in reader.frames(start, end):
            encoder.stdin.write(frame.image.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg could not write {path}")
    return path
```

In `tests/tvd/test_tvd_cli.py`:

Before:

```python
import json

from tvd.cli import main
```

add:

```python
import importlib.util
```

After:

```python
import importlib.util
import json
```

add:

```python
from pathlib import Path

import pytest
from conftest import ROOT
from notebook_data import NOTEBOOK_08_RLR, RED_LIGHT_SCENE, RLR_VIDEO, cut_losslessly, load_json
```

After:

```python
from notebook_data import NOTEBOOK_08_RLR, RED_LIGHT_SCENE, RLR_VIDEO, cut_losslessly, load_json

from tvd.cli import main
```

add:

```python
from tvd.plates import PLATE_MODELS, BasePlateModels
```

Add to the end of the file:

```python
PLATES_FILE = """
version: 1
models:
  type: test-plate
reading: {frames_per_track: 12, min_frame_gap: 4, vehicle_margin: 0.15, lowest_plate_center: 0.4, min_plate_width: 30, min_chars: 4, min_char_prob: 0.95, min_votes: 2, min_agreement: 0.67}
"""


def review_args(tmp_path, video, recorded_at: str = "2026-11-07T08:00:00+07:00") -> list[str]:
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE)
    (tmp_path / "perception.yaml").write_text(DOWN)
    (tmp_path / "plates.yaml").write_text(PLATES_FILE)
    return [
        "review", "--video", str(video), "--camera", str(tmp_path / "camera.yaml"), "--recorded-at", recorded_at, "--output", str(tmp_path / "job"),
        "--perception", str(tmp_path / "perception.yaml"), "--plates", str(tmp_path / "plates.yaml"),
    ]


def test_review_writes_a_folder_for_review_and_prints_a_summary(tmp_path, capsys, sample_video):
    assert main(review_args(tmp_path, sample_video)) == 0
    printed = capsys.readouterr().out
    assert "red_light: 1 candidates for review" in printed and f"1 records for review in {tmp_path / 'job'} (duplicates dropped: 0); plates: unreadable 1" in printed
    record = json.loads((tmp_path / "job" / "records.json").read_text())["events"][0]
    assert record["time"]["trigger_at"] == "2026-11-07T08:00:02.917+07:00"  # Frame 70, at 24 frames per second.
    assert set(record["config_files"]) == {"perception", "camera", "rules", "plates", "evidence"} and record["settings"]["rule"]["pass_margin"] == 0.2


def test_review_needs_the_time_zone_of_the_first_frame_before_any_work(tmp_path, capsys, sample_video):
    assert main(review_args(tmp_path, sample_video, recorded_at="2026-11-07T08:00:00")) == 2
    assert "error: recorded_at needs a time zone" in capsys.readouterr().err and not (tmp_path / "job").exists()


def test_review_refuses_a_folder_in_use(tmp_path, capsys, sample_video):
    argv = review_args(tmp_path, sample_video)
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "records.json").write_text("{}")
    assert main(argv) == 2
    assert "error: The output must be a new or empty folder" in capsys.readouterr().err


@PLATE_MODELS.register("test-cannot-start")
class CannotStart(BasePlateModels):
    def __init__(self) -> None:
        raise RuntimeError("Plate models: no such model")

    def detect(self, image):
        return []

    def recognize(self, plate):
        raise NotImplementedError


def test_review_keeps_the_records_when_the_plate_models_cannot_start(tmp_path, capsys, sample_video):
    argv = review_args(tmp_path, sample_video)
    (tmp_path / "plates.yaml").write_text(PLATES_FILE.replace("test-plate", "test-cannot-start"))
    assert main(argv) == 0
    assert "failure: plate reading: the plate models did not start: Plate models: no such model" in capsys.readouterr().out
    assert json.loads((tmp_path / "job" / "records.json").read_text())["events"][0]["license_plate"]["status"] == "not_attempted"


needs_gpu_rlr_and_notebook_08 = pytest.mark.skipif(
    any(importlib.util.find_spec(n) is None for n in ("ultralytics", "boxmot", "onnxruntime", "fast_plate_ocr", "open_image_models"))
    or not RLR_VIDEO.is_file() or not (ROOT / "weights" / "yolo26m.pt").is_file() or not NOTEBOOK_08_RLR.is_dir(),
    reason="Needs the models, the weights, data/raw/rlr_occasions.mp4 and the outputs of Notebook 08, as on the GPU server.",
)


def comparable(record: dict, frame_offset: int) -> dict:
    """What both versions of a record say about a case, with frames counted from the start of the clip."""
    return {
        "violation_type": record["violation_type"], "track_id": record["track"]["track_id"], "class_name": record["track"]["class_name"],
        "trigger_frame": record["time"]["trigger_frame"] - frame_offset, "rule": {k: record["rule"][k] for k in ("name", "version")},
        "why": record["why"], "signal_source": record["signal_source"], "measurements": record["measurements"],
        "license_plate": record["license_plate"], "needs_attention": record["needs_attention"], "review": record["review"],
        "evidence_frames": [(f["role"], f["frame_in_clip"]) for f in record["evidence"]["frames"]],
    }


@needs_gpu_rlr_and_notebook_08
def test_review_reproduces_notebook_08_on_clip_c_cut_into_its_own_file(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    clip = load_json(RED_LIGHT_SCENE)["clips"]["C"]
    video = cut_losslessly(RLR_VIDEO, clip["start"], clip["end"], tmp_path / "rlr_occasions-C.mkv")
    try:
        argv = ["review", "--video", str(video), "--camera", "configs/cameras/rlr_occasions-C.yaml", "--recorded-at", "2026-11-07T20:00:00+07:00", "--output", str(tmp_path / "job")]
        assert main(argv) == 0
    finally:
        video.unlink()  # Real footage is not left behind.
    ours = json.loads((tmp_path / "job" / "records.json").read_text())["events"]
    theirs = [r for r in load_json(NOTEBOOK_08_RLR / "events" / "rlr_occasions_review_events.json")["events"] if r["track"]["clip"] == "C"]
    same = [comparable(r, 0) for r in ours] == [comparable(r, clip["start"]) for r in theirs]  # Outside the assert: a failure never prints a plate.
    assert same and len(ours) == 2
    clips = NOTEBOOK_08_RLR / "violations" / "rlr_occasions_violation_clips"
    assert all((tmp_path / "job" / o["evidence"]["clip"]).read_bytes() == (clips / Path(t["evidence"]["clip"]).name).read_bytes() for o, t in zip(ours, theirs))
```

`tests/tvd/test_tvd_privacy.py`:

```python
"""Real plate numbers must never reach a file that goes to GitHub. The private labels hold some; records and plates/ hold others."""

import json

import pytest
from conftest import ROOT

LABELS = ROOT / "data" / "private_annotations" / "plate_labels.json"
COMMITTED = ("src/tvd", "tests/tvd", "configs", "docs")


@pytest.mark.skipif(not LABELS.is_file(), reason="The private labels are only on machines that have them.")
def test_no_real_plate_appears_in_the_service_its_tests_settings_or_documents():
    videos = json.loads(LABELS.read_text())["videos"].values()
    real = {track["text"] for video in videos for track in video["tracks"].values() if "text" in track}
    files = [p for folder in COMMITTED for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".md", ".json"}]
    found = sorted(str(p.relative_to(ROOT)) for p in files if any(text in p.read_text(errors="ignore") for text in real))
    has_labels = len(real) > 0  # Asserted apart from the set: a failure shows file names, never a plate.
    assert has_labels and found == []
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_cli.py tests/tvd/test_tvd_privacy.py -q
```

Expected: `5 failed, 5 passed`: the five new tests of `test_tvd_cli.py` stop with `SystemExit: 2`, after `tvd: error: argument command: invalid choice: 'review' (choose from 'track', 'check')`.

- [ ] **Step 3: Write the code**

`src/tvd/cli.py`, replaced as a whole:

```python
"""Command line.

- ``python -m tvd.cli track --video <file> --config <yaml> --output <json>``: detect and track.
- ``python -m tvd.cli check --video <file> --camera <yaml> --output <json>``: apply the camera's rules.
- ``python -m tvd.cli review --video <file> --camera <yaml> --recorded-at <time> --output <folder>``: and write
  each candidate's evidence and record, for a person to review.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tvd.camera import load_camera, load_rule_defaults
from tvd.check import check_frame_size, check_video
from tvd.config import PerceptionConfig, file_version, load_config
from tvd.evidence import load_evidence_settings
from tvd.factory import build_detector, build_motion_estimator, build_plate_models, build_rules, build_tracker
from tvd.perception import track_video
from tvd.plates import load_plates_config
from tvd.records import RecordContext, parse_recorded_at
from tvd.review import ReviewResult, prepare_folder, review_check
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader

PERCEPTION = Path("configs/tvd/perception.yaml")
RULE_DEFAULTS = Path("configs/tvd/rules.yaml")
PLATES = Path("configs/tvd/plates.yaml")
EVIDENCE = Path("configs/tvd/evidence.yaml")


def tracks_as_json(history: TrackHistory) -> list[dict]:
    """Each track with its class and its points as ``[frame, x1, y1, x2, y2, confidence]``."""
    return [
        {
            "track_id": track.track_id,
            "class_name": track.class_name,
            "points": [[p.frame_index, *p.box.as_list(), p.confidence] for p in track.points],
        }
        for track in history.tracks.values()
    ]


def run_track(video: Path, config_path: Path, output: Path) -> dict:
    config = load_config(config_path, PerceptionConfig)
    with VideoReader(video) as reader:  # A missing or unreadable video fails here, before any model is loaded.
        detector = build_detector(config.detector)
        tracker = build_tracker(config.tracker)
        start = time.perf_counter()
        history = track_video(reader, detector, tracker, set(config.track_classes))
    summary = {
        "video": video.name,
        "detector": config.detector.type,
        "tracker": config.tracker.type,
        "frames": history.frames_seen,
        "tracks": len(history.tracks),
        "seconds": round(time.perf_counter() - start, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({**summary, "track_list": tracks_as_json(history)}))
    return summary


def run_check(video: Path, camera_path: Path, output: Path, perception_path: Path = PERCEPTION, defaults_path: Path = RULE_DEFAULTS) -> dict:
    """Every settings file is checked, and the video opened, before any model is loaded."""
    perception = load_config(perception_path, PerceptionConfig)
    camera = load_camera(camera_path)
    rules = build_rules(camera, load_rule_defaults(defaults_path))
    motion_estimator = build_motion_estimator(camera.motion)
    with VideoReader(video) as reader:
        check_frame_size(reader, camera)
        detector = build_detector(perception.detector)
        tracker = build_tracker(perception.tracker)
        result = check_video(reader, camera, detector, tracker, set(perception.track_classes), motion_estimator, rules)
    data = result.as_json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=1))
    return data


def check_summary(data: dict) -> str:
    lines = [f"{data['video']}: {data['frames']} frames, {data['tracks']} tracks, camera {data['camera_id']}, {data['seconds']} s"]
    for name, rule in data["rules"].items():
        verdicts = ", ".join(f"{verdict} {count}" for verdict, count in sorted(rule["verdicts"].items()))
        lines.append(f"  {name}: {len(rule['findings'])} candidates for review; {verdicts or 'no tracks judged'}")
    return "\n".join(lines)


def run_review(
    video: Path, camera_path: Path, recorded_at: str, output: Path, perception_path: Path = PERCEPTION,
    defaults_path: Path = RULE_DEFAULTS, plates_path: Path = PLATES, evidence_path: Path = EVIDENCE,
) -> ReviewResult:
    """Every settings file, the time and the output folder are checked, and the video opened, before any model is loaded.

    Plate models that cannot start do not stop the review: its records say that plates were not read, and why.
    """
    moment = parse_recorded_at(recorded_at)
    perception = load_config(perception_path, PerceptionConfig)
    camera = load_camera(camera_path)
    rules = build_rules(camera, load_rule_defaults(defaults_path))
    motion_estimator = build_motion_estimator(camera.motion)
    plates = load_plates_config(plates_path)
    evidence = load_evidence_settings(evidence_path)
    files = {"perception": perception_path, "camera": camera_path, "rules": defaults_path, "plates": plates_path, "evidence": evidence_path}
    config_files = {name: file_version(path) for name, path in files.items()}
    with VideoReader(video) as reader:
        check_frame_size(reader, camera)
        prepare_folder(output)
        try:
            plate_models, unavailable = build_plate_models(plates.models), ""
        except RuntimeError as error:
            plate_models, unavailable = None, f"the plate models did not start: {error}"
        try:
            detector, tracker = build_detector(perception.detector), build_tracker(perception.tracker)
            check = check_video(reader, camera, detector, tracker, set(perception.track_classes), motion_estimator, rules)
            settings = {
                "perception": perception.model_dump(), "rules": {name: r.settings for name, r in check.results.items()},
                "plates": plates.model_dump(), "evidence": evidence.model_dump(),
            }
            context = RecordContext(
                video.stem, reader.fps, reader.width, reader.height, camera.camera_id, moment, f"{video.stem}/{perception.tracker.type}", settings, config_files,
            )
            return review_check(reader, check, context, plate_models, plates.reading, evidence, output, unavailable or "no plate models")
        finally:
            if plate_models is not None:
                plate_models.close()


def review_summary(result: ReviewResult) -> str:
    """Counts only: plate numbers are never printed."""
    plates = ", ".join(f"{status} {count}" for status, count in sorted(result.summary()["plates"].items())) or "none"
    lines = [
        check_summary(result.check.as_json()),
        f"  {len(result.records)} records for review in {result.folder} (duplicates dropped: {len(result.duplicates)}); plates: {plates}",
    ]
    return "\n".join(lines + [f"  failure: {failure}" for failure in result.failures])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tvd")
    commands = parser.add_subparsers(dest="command", required=True)
    track = commands.add_parser("track", help="Detect and track the vehicles of a video")
    track.add_argument("--video", type=Path, required=True)
    track.add_argument("--config", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="Apply a camera's rules to a video of that camera")
    check.add_argument("--video", type=Path, required=True)
    check.add_argument("--camera", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--perception", type=Path, default=PERCEPTION)
    check.add_argument("--rules", type=Path, default=RULE_DEFAULTS)
    review = commands.add_parser("review", help="Apply a camera's rules to a video, and write each candidate's evidence and record")
    review.add_argument("--video", type=Path, required=True)
    review.add_argument("--camera", type=Path, required=True)
    review.add_argument("--recorded-at", required=True, help="The time of the first frame, with its time zone: 2026-11-07T08:00:00+07:00")
    review.add_argument("--output", type=Path, required=True, help="A new or empty folder")
    review.add_argument("--perception", type=Path, default=PERCEPTION)
    review.add_argument("--rules", type=Path, default=RULE_DEFAULTS)
    review.add_argument("--plates", type=Path, default=PLATES)
    review.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            summary = run_track(args.video, args.config, args.output)
            print(f"{summary['video']}: {summary['frames']} frames, {summary['tracks']} tracks, detector {summary['detector']}, tracker {summary['tracker']}, {summary['seconds']} s")
        elif args.command == "check":
            print(check_summary(run_check(args.video, args.camera, args.output, args.perception, args.rules)))
        else:
            result = run_review(args.video, args.camera, args.recorded_at, args.output, args.perception, args.rules, args.plates, args.evidence)
            print(review_summary(result))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:  # ValueError covers ConfigError.
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`configs/cameras/rlr_occasions-C.yaml`:

```yaml
# Clip C of data/raw/rlr_occasions.mp4 (frames 914 to 1121), cut into its own file: a fixed camera at night.
# The source is a compilation, so each of its clips is a camera of its own. From clip C of configs/scenes/rlr_occasions.json.
camera_id: rlr_occasions-C
frame_size: [1280, 720]
rules:
  red_light:
    scene:
      stop_line: [[190, 490], [420, 492], [664, 500], [936, 508], [1180, 522], [1280, 532]]
      lane_dividers: [420, 664, 936, 1180]    # left of 420: the left-turn lane; right of 1180: the right-turn lane
      lane_direction_deg: -24.0
      traffic_direction: away_from_camera
      evaluable_x_max: 1280
      signals:
        reader: glow                          # at night only the lamp's glow can be read
        through:                              # mast heads 2 and 3
          - {roi: [835, 80, 878, 150]}
          - {roi: [985, 100, 1026, 170]}
        left:                                 # mast head 1
          - {roi: [650, 62, 685, 128]}
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `263 passed`.

- [ ] **Step 5: Review a whole video**

The camera moves, so camera motion takes most of the time, as in Part 2.

```bash
scripts/server.sh run python -m tvd.cli review --video data/traffic_video_modified.mp4 --camera configs/cameras/traffic_video_modified.yaml \
    --recorded-at 2026-11-07T19:30:00+07:00 --output outputs/tvd/review/traffic_video_modified
```

Expected, after some log lines from BoxMOT, in about 2 minutes (Notebook 08 took almost 4; the seconds vary):

```text
traffic_video_modified.mp4: 715 frames, 45 tracks, camera traffic_video_modified, 107.2 s
  red_light: 2 candidates for review; candidate 2, not_evaluable_signal 1, signal_not_red 7, stopped_at_line 1
  2 records for review in outputs/tvd/review/traffic_video_modified (duplicates dropped: 0); plates: read 2
```

The folder holds `records.json`, `summary.json`, and `clips/`, `evidence/` and `plates/`. `records.json` and `plates/` show real plate numbers: do not print them, and they stay on the server, under `outputs/`. Run the command again and it stops with `error: The output must be a new or empty folder`.

- [ ] **Step 6: Compare with Notebook 08**

Each clip and plate image must be the same file as Notebook 08's:

```bash
scripts/server.sh run 'for f in f00317-t23 f00365-t37; do cmp outputs/tvd/review/traffic_video_modified/clips/evt-traffic_video_modified-red_light-$f.mp4 outputs/08_end_to_end_baseline/traffic_video_modified/violations/traffic_video_modified_violation_clips/evt-traffic_video_modified-main-red_light-$f.mp4 && echo "clip $f: same"; done'
scripts/server.sh run 'cd outputs && cmp tvd/review/traffic_video_modified/plates/evt-traffic_video_modified-red_light-f00317-t23_f00116.png 08_end_to_end_baseline/traffic_video_modified/plates/plate_crops/evt-traffic_video_modified-main-red_light-f00317-t23_f00116.png && echo "plate image: same"'
```

Expected: `clip f00317-t23: same`, `clip f00365-t37: same` and `plate image: same`. The ids differ only by `-main-`, the name of the notebook's clip.

- [ ] **Step 7: Look at one evidence image**

Copy one image to your laptop and open it. It shows a real plate: keep it out of the repository.

```bash
scp 20.0.14.182:workspace/dungvu/traffic-violation-detection-system/outputs/tvd/review/traffic_video_modified/evidence/evt-traffic_video_modified-red_light-f00317-t23_entry.jpg /tmp/
```

Expected: the stop line in magenta across the road, a yellow box around the car on the line, and at the top left, on a dark band, `through signal: red` and `left signal: red` in red, and `signal source: simulated_overlay` in white. Delete the copy when you are done.

- [ ] **Step 8: Commit**

```bash
git add tests/tvd/notebook_data.py tests/tvd/test_tvd_cli.py tests/tvd/test_tvd_privacy.py src/tvd/cli.py configs/cameras/rlr_occasions-C.yaml
git commit -m "Add the review command"
```

---

## When Part 3 is done

- [ ] `scripts/server.sh test tests/tvd -q` gives `263 passed`, and `scripts/server.sh test -q` still passes the notebooks' tests too: `732 passed` in all.
- [ ] `review` on `traffic_video_modified.mp4` gives two records, with plates `read 2`, and the same clips and plate images as Notebook 08 (Task 12, Steps 5 and 6).
- [ ] A red-light evidence image shows the stop line on the road and the state of each signal.
- [ ] Stopping the plate models, or breaking `plates.yaml`'s model name, still gives every record, with `not_attempted`.
- [ ] No plate number was printed in your terminal or committed: `git status` shows nothing under `outputs/`.

**Exercises, if you want them** (not checked while writing the plan):

- **An overlay for wrong way**: a `WrongWayOverlay` that draws each region's polygon and an arrow of its allowed direction, returned in the wrong-way rule's `RuleResult`. `review_check` does not change.
- **A plate model that runs in this process**, for a machine without PyTorch: a class that creates the models directly and is registered as `onnx_local`. Choose it in `plates.yaml` and check that every plate test still passes.

**Next:** Part 4, the service, puts a job queue, storage and a REST API around `check_video` and `review_check`: the plate models load once when the service starts, and each upload gets a new job folder. Its plan is written once Part 3 is done.
