# Service Part 2: Scene and Rules — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the wrong-way and red-light rules as replaceable components, with every camera described in its own YAML file and every rule setting in another, and add a `check` command that applies a camera's rules to a video in one pass.

**Architecture:** Part 1's `track_video` now feeds **frame observers**: the red-light rule's signal monitor, and a log of detections for a moving camera. After the pass, a **motion estimator** (fixed, or ORB for a moving camera) maps each frame to the reference frame the camera file is drawn on, and every **rule** judges the same `Observation`. Signal readers, motion estimators and rules each have a base class and a registry, as detectors and trackers have in Part 1. Tests compare every rule with the notebooks' code on the notebooks' own saved data.

**Tech Stack:** Python 3.11, OpenCV 5.0.0, NumPy 2.4.6, SciPy, pydantic 2, PyYAML 6, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-07-review-service-design.md` (Sections 4, 5 and 12).

**Verified:** every file below, the state before each task's code (with the failure it shows) and the state after it were run on ai_server_2 before this plan was written. After the last task: 144 tests pass in `tests/tvd`, and `check` gives the outcomes of Notebook 08 on `traffic_video_modified.mp4` and of Notebook 05 on `video-test.mp4`.

## The five parts

| Part | Builds | Plan |
|---|---|---|
| 1. Foundations and perception | Data types, registry, video reader, detector, trackers, settings, `track` command | Done: `2026-11-07-service-part-1-foundations-and-perception.md` |
| **2. Scene and rules** | Camera files, rule settings, camera motion, signal reading, the wrong-way and red-light rules, a `check` command | This document |
| 3. Plates and evidence | Plate detection and OCR in a worker process, voting, clips, images with the stop line and signal state, records | Later |
| 4. The service | Job queue, storage, REST API with FastAPI, around `check_video` | Later |
| 5. Deployment and MLOps | Docker, running on ai_server_2, logging, health checks, model versions | Later |

## What one `check` does

```text
video ──> VideoReader ──> for each frame: detect ──> observers (SignalMonitor, DetectionLog) ──> track
                                                         │
camera file ──> motion estimator ("fixed" | "orb") ──> CameraMotion: frame pixels -> reference pixels
                                                         │
                         Observation (tracks, motion, size, fps, frame range)
                                   │                          │
                         WrongWayRule.evaluate      RedLightRule.evaluate (its SignalMonitor's states)
                                   │                          │
                         RuleResult: findings for review, one record per track, the settings used
```

A **finding** is a candidate for a person to review, never a verdict, as in the notebooks.

## Global Constraints

- Python 3.11, in the `tvd` environment: on ai_server_2, `~/workspace/dungvu/envs/tvd`. No new dependency in this part.
- Never let pip change OpenCV (5.0.0.93) or NumPy (2.4.6).
- New code goes in `src/tvd/`, and its tests in `tests/tvd/test_tvd_*.py`. The notebooks' package `traffic_violation_detection` does not change: it is the reference.
- **Nothing that varies between cameras or that a user might tune is written in code.** Each camera's geometry is in `configs/cameras/<camera_id>.yaml`; every rule setting is in `configs/tvd/rules.yaml`, and a camera file may override any of them for that camera under `settings`. The rule settings models have no defaults, so a setting missing from the files is an error, not a silent value.
- Notebook results are the reference. Where a test compares with them, the new code must match **exactly**: the same verdict for every track, the same records, the same candidates.
- Tests that need the GPU stack, `data/raw/rlr_occasions.mp4`, or the saved outputs of Notebooks 03, 04 and 05b skip themselves where those are missing. The counts below are from the GPU server, where nothing skips.

## Review Focus

- **A video from another camera** would put the stop line in the wrong place. A frame size that differs from the camera file's is refused before any model loads: `test_a_video_of_another_size_is_refused` (Task 14), `test_check_refuses_a_video_of_another_camera` (Task 15).
- **A misspelt or out-of-range setting in a camera file** must name the rule and the key, not be ignored: `test_a_misspelt_camera_setting_is_named` (Task 13), `test_a_wrong_setting_is_named` (Tasks 9 and 11).
- **`settings:` or `options:` left empty in YAML** reads as null, and must mean "none": `test_settings_left_empty_mean_none` (Task 12); `SignalHeads` treats `options` the same way (Task 11).
- **A signal group without heads**, such as no left-turn signal, must read `unknown` and make its crossings not evaluable, never red: `test_a_monitor_smooths_each_group_and_leaves_a_group_without_heads_unknown` (Task 7), `test_the_left_turn_lane_follows_the_left_signal` (Task 11).
- **A moving camera's reference frame outside the analysed frames** has nothing to compare with: refused by `test_the_reference_frame_must_be_analysed` (Task 5).

## How you will work

As in Part 1: edit on your laptop, run on ai_server_2 with `scripts/server.sh`. After Task 0, `sync` makes the server's `src/`, `tests/`, `configs/`, `scripts/` and `pyproject.toml` the same as yours, deleting what you deleted.

| Command | Does |
|---|---|
| `scripts/server.sh sync` | Makes the server's copy the same as here |
| `scripts/server.sh test tests/tvd -q` | Runs pytest there |
| `scripts/server.sh test tests/tvd -q -k "red and notebooks"` | Runs only the tests whose names match |
| `scripts/server.sh run <command>` | Runs any command there, in the project folder, with the `tvd` environment |

Several tests read what the notebooks saved on the server: `outputs/03_multi_object_tracking/tracks/video-test.ocsort-byte.jsonl` (tracks), `outputs/04_traffic_scene_configuration/video-test.camera_motion.json` (camera motion), and `outputs/05b_red_light_violation/tracks.json` and `signals.json` (the tracks and signal states of 18 clips). They give both implementations the same input, so the comparison is exact and needs no GPU.

## File structure

```text
src/tvd/
├── observers.py              Task 2   FrameObserver, DetectionLog
├── perception.py             Task 2   track_video(..., observers)                      (changed)
├── geometry.py               Task 3   polyline_y, point_in_polygon
├── motion/
│   ├── __init__.py           Task 4/5 exports; importing it registers the estimators
│   ├── base.py               Task 4   CameraMotion, BaseMotionEstimator, MOTION_ESTIMATORS
│   ├── fixed.py              Task 4   FixedCamera  ("fixed")
│   └── orb.py                Task 5   OrbMotionEstimator  ("orb")
├── signals/
│   ├── __init__.py           Task 6/7 exports; importing it registers the readers
│   ├── base.py               Task 6   BaseSignalReader, SIGNAL_READERS, combine_states, smooth_states
│   ├── lamps.py              Task 6   LampReader  ("lamps", by day)
│   ├── glow.py               Task 6   GlowReader  ("glow", at night)
│   └── monitor.py            Task 7   SignalMonitor, SignalTimeline
├── rules/
│   ├── __init__.py           Task 8/9/11 exports; importing it registers the rules
│   ├── base.py               Task 8   Observation, Finding, RuleResult, BaseRule, RULES
│   ├── wrong_way.py          Task 9   WrongWayRule  ("wrong_way")
│   └── red_light.py          Task 11  RedLightRule  ("red_light")
├── tracking/
│   ├── history.py            Task 1   TrackPoint gains class_name; majority_class       (changed)
│   └── fragments.py          Task 10  join_fragments
├── config.py                 Task 8   describe, validate                                (changed)
├── camera.py                 Task 12  CameraConfig, RuleSection, load_camera, load_rule_defaults
├── factory.py                Task 13  build_motion_estimator, build_rules               (changed)
├── check.py                  Task 14  check_video, CheckResult
└── cli.py                    Task 15  python -m tvd.cli check ...                       (changed)
configs/tvd/rules.yaml        Task 9/11  every rule setting
configs/cameras/*.yaml        Task 12  video-test (wrong way), traffic_video_modified (red light)
scripts/server.sh             Task 0   sync deletes; test keeps arguments whole          (changed)
tests/tvd/notebook_data.py    Task 7   turns the notebooks' scenes and saved data into tvd's inputs
```

**How the parts depend on each other.** Arrows point at what a module uses. Only `factory.py` and `cli.py` know concrete classes, by name; a rule builds its own signal readers by name, from its scene.

```text
cli ──> factory ──> registry <── motion/{fixed,orb}, signals/{lamps,glow}, rules/{wrong_way,red_light}
 │        └──> camera ──> config
 └──> check ──> perception ──> observers
         ├──> motion/base
         └──> rules/base ──> motion/base, observers, tracking/history
rules/red_light ──> signals, tracking/fragments, geometry        rules/wrong_way ──> geometry
```

## SOLID and patterns in this part

| Principle or pattern | Where |
|---|---|
| **S**ingle responsibility | `geometry.py` computes, `signals/` reads lamps, `motion/` maps frames, each rule judges one kind of violation, `camera.py` reads camera files, `check.py` runs the pass |
| **O**pen/closed | A new rule, signal reader or motion estimator is a new class with `@RULES.register`, `@SIGNAL_READERS.register` or `@MOTION_ESTIMATORS.register`; a new kind of per-frame work is a new `FrameObserver`. `check_video` and the camera file format do not change |
| **L**iskov substitution | `FixedCamera` and `OrbMotionEstimator` both return a `CameraMotion`; `LampReader`, `GlowReader` and the test reader all return a state; both rules return a `RuleResult` |
| **I**nterface segregation | `BaseSignalReader` has one method; `BaseRule` one required method and one optional; `FrameObserver` one method |
| **D**ependency inversion | `check_video` takes built components; rules receive an `Observation`, never a video or a file |
| Observer | `FrameObserver`: the pass calls every observer on every frame |
| Strategy | Signal readers, motion estimators and rules, chosen by name in a YAML file |

---

### Task 0: Branch, and a server script that deletes and quotes

**Why:** Part 2 adds and renames many files. The old `sync` only copied, so a test file deleted or renamed on your laptop stayed on the server and kept running. `--delete` makes the server's copy exact, and `printf %q` passes a pytest argument with spaces, such as `-k "yolo and notebooks"`, as one argument.

**Files:**
- Replace: `scripts/server.sh`

**Interfaces:**
- Consumes: `scripts/server.sh` of Part 1.
- Produces: `scripts/server.sh sync` that also deletes, and `scripts/server.sh test` that keeps each argument whole.

- [ ] **Step 1: Create a branch from `main`**

```bash
git checkout main && git checkout -b service/part-2
```

- [ ] **Step 2: Replace the server script**

`scripts/server.sh`:

```bash
#!/usr/bin/env bash
# Work on the GPU server from this machine.
#   scripts/server.sh sync             make the server's src/, tests/, configs/, scripts/ and pyproject.toml the same as here
#   scripts/server.sh test [args]      run pytest there, e.g. scripts/server.sh test tests/tvd -q -k "yolo and notebooks"
#   scripts/server.sh run <command>    run a command there, in the project folder, with the tvd environment first on PATH
set -euo pipefail
HOST="${TVD_SERVER:-20.0.14.182}"
REMOTE_DIR="${TVD_REMOTE_DIR:-workspace/dungvu/traffic-violation-detection-system}"
REMOTE_ENV="${TVD_REMOTE_ENV:-workspace/dungvu/envs/tvd}"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

remote() {
    "${SSH[@]}" "$HOST" "cd ~/$REMOTE_DIR && export PATH=~/$REMOTE_ENV/bin:\$PATH && $*"
}

case "${1:-}" in
    sync)
        # --delete removes files deleted or renamed here, so the server never runs a test that no longer exists.
        # Excluded files are neither copied nor deleted: each machine keeps its own caches and install records.
        rsync -az --delete --timeout=60 -e "${SSH[*]}" --exclude '__pycache__/' --exclude '*.egg-info/' \
            src tests configs scripts pyproject.toml "$HOST:$REMOTE_DIR/"
        echo "synced to $HOST:$REMOTE_DIR"
        ;;
    test)
        shift
        # printf %q quotes each argument again, so that one with spaces reaches pytest as one argument.
        args=""
        if [ $# -gt 0 ]; then args=$(printf '%q ' "$@"); fi
        remote python -m pytest "$args"
        ;;
    run)
        shift
        remote "$@"
        ;;
    *)
        echo "usage: scripts/server.sh sync | test [pytest args] | run <command>" >&2
        exit 2
        ;;
esac
```

- [ ] **Step 3: Before the first sync, see what it would delete**

`--delete` removes from the server's `src/`, `tests/`, `configs/` and `scripts/` whatever is not on your laptop. Check once that this is nothing you need (`-n` only shows what would happen):

```bash
rsync -azn --delete -i -e ssh --exclude '__pycache__/' --exclude '*.egg-info/' src tests configs scripts pyproject.toml 20.0.14.182:workspace/dungvu/traffic-violation-detection-system/ | grep deleting
```

Expected: no output.

- [ ] **Step 4: Sync, and pass an argument with spaces**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q -k "yolo and notebooks"
```

Expected: `1 passed, 53 deselected`. The script of Part 1 gave `ERROR: file or directory not found: and`, because pytest received `and` and `notebooks` as file names.

- [ ] **Step 5: Run every test of the package, as before**

```bash
scripts/server.sh test tests/tvd -q
```

Expected: `54 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/server.sh
git commit -m "Make the server copy exact, and pass test arguments as given"
```

### Task 1: Each track point keeps its class

**Why:** The red-light rule joins two tracks of one vehicle, and the joined track's class must be the majority over the joined points. So each point now keeps the class the detector gave it, and `majority_class` works on any list of points. **Single source of truth:** the per-track counter of Part 1 is gone; the class is computed from the points.

**Files:**
- Replace: `tests/tvd/test_tvd_history.py`
- Replace: `src/tvd/tracking/history.py`

**Interfaces:**
- Consumes: `TrackedObject` (Part 1).
- Produces: `TrackPoint(frame_index, box, confidence, class_name)`; `majority_class(points) -> str`; `Track.class_name` as before.

- [ ] **Step 1: Write the failing test**

`test_tvd_history.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/test_tvd_history.py`:

```python
from tvd.domain import Box, TrackedObject
from tvd.tracking.history import TrackHistory, majority_class


def tracked(track_id: int, x: float, class_name: str = "car") -> TrackedObject:
    return TrackedObject(track_id, Box(x, 0, x + 10, 10), 2, class_name, 0.9)


def test_objects_are_grouped_by_track_id():
    history = TrackHistory()
    history.add(0, [tracked(1, 0), tracked(2, 50)])
    history.add(1, [tracked(1, 5)])
    tracks = history.tracks
    assert sorted(tracks) == [1, 2] and history.frames_seen == 2
    assert (tracks[1].first_frame, tracks[1].last_frame, len(tracks[1].points)) == (0, 1, 2)


def test_a_frame_with_nothing_tracked_still_counts():
    history = TrackHistory()
    history.add(0, [])
    assert history.frames_seen == 1 and history.tracks == {}


def test_the_class_is_the_one_reported_most():
    history = TrackHistory()
    for index, name in enumerate(["car", "truck", "truck"]):
        history.add(index, [tracked(1, 0, name)])
    assert history.tracks[1].class_name == "truck"


def test_a_tie_keeps_the_class_reported_first():
    history = TrackHistory()
    history.add(0, [tracked(1, 0, "car")])
    history.add(1, [tracked(1, 0, "truck")])
    assert history.tracks[1].class_name == "car"


def test_box_at_a_frame():
    history = TrackHistory()
    history.add(3, [tracked(1, 7)])
    assert history.tracks[1].box_at(3) == Box(7, 0, 17, 10) and history.tracks[1].box_at(4) is None


def test_each_point_keeps_its_class():
    history = TrackHistory()
    history.add(0, [tracked(1, 0, "car")])
    history.add(1, [tracked(1, 0, "truck")])
    assert [p.class_name for p in history.tracks[1].points] == ["car", "truck"]


def test_majority_class_of_any_points():
    history = TrackHistory()
    for index, name in enumerate(["bus", "car", "car"]):
        history.add(index, [tracked(1, 0, name)])
    assert majority_class(history.tracks[1].points[:1]) == "bus" and majority_class(history.tracks[1].points) == "car"
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_history.py -q
```

Expected: `E   ImportError: cannot import name 'majority_class' from 'tvd.tracking.history'`

- [ ] **Step 3: Write the code**

`src/tvd/tracking/history.py`:

```python
"""Tracks as they build up over a video."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from tvd.domain import Box, TrackedObject


@dataclass(frozen=True)
class TrackPoint:
    """Where a track was in one frame, and the class the detector gave it there."""

    frame_index: int
    box: Box
    confidence: float
    class_name: str


def majority_class(points: Iterable[TrackPoint]) -> str:
    """The class reported most often; on a tie, the one reported first. A single frame's class can be wrong."""
    return Counter(p.class_name for p in points).most_common(1)[0][0]


@dataclass
class Track:
    """Every frame in which one track id was reported."""

    track_id: int
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def class_name(self) -> str:
        return majority_class(self.points)

    @property
    def first_frame(self) -> int:
        return self.points[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame_index

    def box_at(self, frame_index: int) -> Box | None:
        return next((p.box for p in self.points if p.frame_index == frame_index), None)


class TrackHistory:
    """Collects what a tracker reports, frame by frame, into one ``Track`` per track id."""

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self.frames_seen = 0

    def add(self, frame_index: int, objects: list[TrackedObject]) -> None:
        self.frames_seen += 1
        for tracked in objects:
            track = self._tracks.setdefault(tracked.track_id, Track(tracked.track_id))
            track.points.append(TrackPoint(frame_index, tracked.box, tracked.confidence, tracked.class_name))

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `56 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_history.py src/tvd/tracking/history.py
git commit -m "Keep each track point's class"
```

### Task 2: Frame observers

**Why:** The red-light rule must read the signals in every frame, and a moving camera needs every frame's detections. Instead of a second pass over the video for each, `track_video` hands every frame to a list of **observers** (the observer pattern). A new kind of per-frame work is a new `FrameObserver`, and `track_video` does not change (**open/closed**).

**Files:**
- Replace: `tests/tvd/test_tvd_perception.py`
- Create: `src/tvd/observers.py`
- Replace: `src/tvd/perception.py`

**Interfaces:**
- Consumes: `track_video` (Part 1), `Frame`, `Detection`.
- Produces: `FrameObserver.observe(frame, detections)` (abstract); `DetectionLog` with `by_frame: dict[int, list[Detection]]`; `track_video(..., observers=())`, where observers see all detections, before they are filtered for the tracker.

- [ ] **Step 1: Write the failing test**

`test_tvd_perception.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/test_tvd_perception.py`:

```python
import importlib.util

import pytest
from conftest import ROOT

from tvd.detection import DETECTORS
from tvd.observers import DetectionLog, FrameObserver
from tvd.perception import track_video
from tvd.tracking import TRACKERS
from tvd.video import VideoReader

needs_gpu_stack = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("ultralytics", "boxmot")) or not (ROOT / "weights" / "yolo26m.pt").is_file(),
    reason="Needs Ultralytics, BoxMOT and weights/yolo26m.pt, as on the GPU server.",
)


def test_only_the_chosen_classes_are_tracked(sample_video):
    with VideoReader(sample_video) as reader:
        history = track_video(reader, DETECTORS.create("test-moving-car"), TRACKERS.create("test-same-id"), {"car"}, 0, 5)
    assert history.frames_seen == 5 and list(history.tracks) == [1]
    assert history.tracks[1].class_name == "car" and len(history.tracks[1].points) == 5


def test_the_tracker_is_reset_before_each_video(sample_video):
    tracker = TRACKERS.create("test-same-id")
    with VideoReader(sample_video) as reader:
        track_video(reader, DETECTORS.create("test-moving-car"), tracker, {"car"}, 0, 2)
        track_video(reader, DETECTORS.create("test-moving-car"), tracker, {"car"}, 0, 2)
    assert tracker.resets == 2


def test_observers_see_every_frame_with_all_its_detections(sample_video):
    class FrameCounter(FrameObserver):
        def __init__(self):
            self.frames = []

        def observe(self, frame, detections):
            self.frames.append(frame.index)

    counter, log = FrameCounter(), DetectionLog()
    with VideoReader(sample_video) as reader:
        track_video(reader, DETECTORS.create("test-moving-car"), TRACKERS.create("test-same-id"), {"car"}, 3, 6, observers=[counter, log])
    assert counter.frames == [3, 4, 5] and sorted(log.by_frame) == [3, 4, 5]
    assert [d.class_name for d in log.by_frame[4]] == ["car", "person"]


@needs_gpu_stack
def test_the_same_tracks_as_the_notebooks(monkeypatch):
    from traffic_violation_detection.detection import load_detector
    from traffic_violation_detection.tracking import track_frames

    monkeypatch.chdir(ROOT)
    video, stop = ROOT / "data" / "video-test.mp4", 150
    expected = {
        record["frame_index"]: sorted((t["track_id"], t["bbox_xyxy"]) for t in tracked)
        for record, _, tracked in track_frames(video, load_detector(), 0, stop)
    }
    detector = DETECTORS.create("yolo", weights="yolo26m.pt", classes=["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"])
    with VideoReader(video) as reader:
        history = track_video(reader, detector, TRACKERS.create("ocsort", use_byte=True), {"bicycle", "car", "motorcycle", "bus", "truck"}, 0, stop)
    found = {frame: [] for frame in expected}
    for track in history.tracks.values():
        for point in track.points:
            found[point.frame_index].append((track.track_id, point.box.as_list()))
    assert {frame: sorted(items) for frame, items in found.items()} == expected
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_perception.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.observers'`

- [ ] **Step 3: Write the code**

`src/tvd/observers.py`:

```python
"""Work that must see every frame of the pass over a video, next to detection and tracking."""

from abc import ABC, abstractmethod

from tvd.domain import Detection, Frame


class FrameObserver(ABC):
    """Sees each frame once, with all of its detections, before they are filtered for the tracker.

    Reading the signals or keeping the detections for camera-motion estimation are observers. A new
    kind of per-frame work is a new observer, and the pass over the video does not change (open/closed).
    """

    @abstractmethod
    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        """Called once per frame, in order."""


class DetectionLog(FrameObserver):
    """Every detection of every frame, of every class: what camera-motion estimation masks out of the image."""

    def __init__(self) -> None:
        self.by_frame: dict[int, list[Detection]] = {}

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self.by_frame[frame.index] = list(detections)
```

`src/tvd/perception.py`:

```python
"""Detection and tracking over a video."""

from collections.abc import Sequence

from tvd.detection import BaseDetector
from tvd.observers import FrameObserver
from tvd.tracking import BaseTracker
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader


def track_video(
    reader: VideoReader,
    detector: BaseDetector,
    tracker: BaseTracker,
    track_classes: set[str],
    start: int = 0,
    stop: int | None = None,
    observers: Sequence[FrameObserver] = (),
) -> TrackHistory:
    """Detect every frame from ``start`` to ``stop``, and track the detections of ``track_classes``.

    Each observer sees every frame with all of its detections, so the video is decoded once for all of
    them. It depends only on abstractions, so any detector, tracker and observer work here.
    """
    tracker.reset()
    history = TrackHistory()
    for frame in reader.frames(start, stop):
        detections = detector.detect(frame)
        for observer in observers:
            observer.observe(frame, detections)
        history.add(frame.index, tracker.update([d for d in detections if d.class_name in track_classes], frame))
    return history
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `57 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_perception.py src/tvd/observers.py src/tvd/perception.py
git commit -m "Let observers see every frame of the pass"
```

### Task 3: Geometry

**Why:** Two small pieces of plane geometry that the rules use: the height of a stop line at a column, and whether a point is in a region. **Single responsibility:** pure functions, no state, tested against the notebooks' versions.

**Files:**
- Create: `tests/tvd/test_tvd_geometry.py`
- Create: `src/tvd/geometry.py`

**Interfaces:**
- Consumes: Nothing.
- Produces: `polyline_y(points, x)` (a value or an array); `point_in_polygon(polygon, point) -> bool`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_geometry.py`:

```python
import numpy as np
import pytest

from tvd.geometry import point_in_polygon, polyline_y

LINE = [[0, 100], [100, 110], [300, 150]]
SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_polyline_height_between_its_points():
    assert polyline_y(LINE, 50) == pytest.approx(105) and polyline_y(LINE, 200) == pytest.approx(130)


def test_polyline_height_beyond_its_ends_is_the_nearest_end():
    assert polyline_y(LINE, np.array([-20.0, 400.0])).tolist() == [100.0, 150.0]


@pytest.mark.parametrize("point, inside", [((5, 5), True), ((10, 5), True), ((11, 5), False)])
def test_point_in_polygon(point, inside):
    assert point_in_polygon(SQUARE, point) is inside


def test_the_same_answers_as_the_notebooks():
    from traffic_violation_detection.red_light import line_y
    from traffic_violation_detection.scene import point_in_polygon as notebook_point_in_polygon

    xs = np.linspace(-50, 350, 41)
    assert polyline_y(LINE, xs).tolist() == line_y(LINE, xs).tolist()
    points = [(x, y) for x in range(-2, 13) for y in range(-2, 13)]
    assert [point_in_polygon(SQUARE, p) for p in points] == [notebook_point_in_polygon(SQUARE, p) for p in points]
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_geometry.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.geometry'`

- [ ] **Step 3: Write the code**

`src/tvd/geometry.py`:

```python
"""Plane geometry in pixels, with the origin at the top-left and y pointing down."""

from collections.abc import Sequence

import cv2
import numpy as np

Point = Sequence[float]


def polyline_y(points: Sequence[Point], x):
    """Height of a polyline, such as a stop line, at image column ``x`` (one value or an array).

    Its points must have increasing x. Beyond its ends the height of the nearest end is used.
    """
    xs, ys = np.array(points, dtype=np.float64).T
    return np.interp(x, xs, ys)


def point_in_polygon(polygon: Sequence[Point], point: Point) -> bool:
    """True if the point lies inside the polygon or on its boundary."""
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), (float(point[0]), float(point[1])), False) >= 0
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `63 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_geometry.py src/tvd/geometry.py
git commit -m "Add the geometry the rules need"
```

### Task 4: Camera motion interface and the fixed camera

**Why:** A camera file draws its stop line and regions on one reference frame. When the camera moves, every frame needs its own transform before a vehicle's position can be compared with them. `CameraMotion` holds those transforms; without any, it is a fixed camera and returns points unchanged. `BaseMotionEstimator` is the strategy that finds them, and `FixedCamera` the simplest one: it computes nothing, which is why the spec makes it the default.

**Files:**
- Create: `tests/tvd/test_tvd_motion.py`
- Create: `src/tvd/motion/base.py`
- Create: `src/tvd/motion/fixed.py`
- Create: `src/tvd/motion/__init__.py`

**Interfaces:**
- Consumes: `Registry` (Part 1), `VideoReader`, `Detection`.
- Produces: `CameraMotion(params=None, first_frame=0)` with `is_fixed`, `params_at(frame)`, `to_reference(frame, points)`, `to_frame(frame, points)`; `BaseMotionEstimator.estimate(reader, detections, start, stop) -> CameraMotion` and its flag `needs_detections`; `MOTION_ESTIMATORS`; `FixedCamera` as `"fixed"`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_motion.py`:

```python
import numpy as np
import pytest

from tvd.motion import MOTION_ESTIMATORS, CameraMotion
from tvd.video import VideoReader


def test_the_fixed_camera_is_registered():
    assert "fixed" in MOTION_ESTIMATORS.names()


def test_a_fixed_camera_leaves_points_unchanged(sample_video):
    with VideoReader(sample_video) as reader:
        motion = MOTION_ESTIMATORS.create("fixed").estimate(reader, {}, 0, 10)
    assert motion.is_fixed and motion.to_reference(5, [[12.5, 7.25]]).tolist() == [[12.5, 7.25]]


def test_a_moving_camera_maps_both_ways():
    motion = CameraMotion(np.array([[1.0, 0.0, 10.0, -5.0], [0.0, 1.0, 0.0, 0.0]]), first_frame=100)
    assert motion.to_reference(100, [[0, 0]]).tolist() == [[10.0, -5.0]]
    assert motion.to_reference(101, [[2, 0]])[0].tolist() == pytest.approx([0.0, 2.0])  # A quarter turn.
    assert motion.to_frame(100, motion.to_reference(100, [[3, 4]]))[0].tolist() == pytest.approx([3.0, 4.0])


def test_a_frame_outside_the_estimate_is_refused():
    with pytest.raises(IndexError, match="Frame 102 is outside"):
        CameraMotion(np.zeros((2, 4)), first_frame=100).params_at(102)
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_motion.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.motion'`

- [ ] **Step 3: Write the code**

`src/tvd/motion/base.py`:

```python
"""Camera motion: how each frame's pixels map to the pixels of one reference frame.

A camera file draws its stop line and regions on the reference frame. When the camera moves, each
frame needs its own transform to compare a vehicle's position with them. The transform of a frame is a
similarity ``(a, b, tx, ty)``: ``x' = a*x - b*y + tx``, ``y' = b*x + a*y + ty``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import cv2
import numpy as np

from tvd.domain import Detection
from tvd.registry import Registry
from tvd.video import VideoReader

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def similarity_matrix(params: Sequence[float]) -> np.ndarray:
    """The 2x3 matrix of a similarity transform given as ``(a, b, tx, ty)``."""
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


class CameraMotion:
    """The transform of every analysed frame, from frame pixels to reference pixels.

    Built without ``params``, it is a fixed camera: points are returned unchanged, with no arithmetic,
    so that a fixed camera's results never depend on rounding.
    """

    def __init__(self, params: np.ndarray | None = None, first_frame: int = 0) -> None:
        self._params = None if params is None else np.asarray(params, dtype=np.float64).reshape(-1, 4)
        self.first_frame = first_frame

    @property
    def is_fixed(self) -> bool:
        return self._params is None

    def params_at(self, frame_index: int) -> np.ndarray:
        if self._params is None:
            return np.array(IDENTITY)
        offset = frame_index - self.first_frame
        if not 0 <= offset < len(self._params):
            raise IndexError(f"Frame {frame_index} is outside the {len(self._params)} frames from {self.first_frame}")
        return self._params[offset]

    def to_reference(self, frame_index: int, points) -> np.ndarray:
        """``(x, y)`` points of a frame, in reference pixels."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._params is None:
            return points
        return cv2.transform(points[None], similarity_matrix(self.params_at(frame_index)))[0]

    def to_frame(self, frame_index: int, points) -> np.ndarray:
        """Reference points, in the pixels of a frame: where to draw a stop line on that frame."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._params is None:
            return points
        inverse = cv2.invertAffineTransform(similarity_matrix(self.params_at(frame_index)))
        return cv2.transform(points[None], inverse)[0]


class BaseMotionEstimator(ABC):
    """Finds how the camera moved in each frame of a range."""

    needs_detections: bool = False
    """True if ``estimate`` needs every frame's detections, to leave moving objects out."""

    @abstractmethod
    def estimate(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> CameraMotion:
        """The motion of frames ``start`` to ``stop`` (not included)."""


MOTION_ESTIMATORS: Registry[BaseMotionEstimator] = Registry("camera motion estimator")
```

`src/tvd/motion/fixed.py`:

```python
"""A camera that does not move."""

from tvd.domain import Detection
from tvd.motion.base import MOTION_ESTIMATORS, BaseMotionEstimator, CameraMotion
from tvd.video import VideoReader


@MOTION_ESTIMATORS.register("fixed")
class FixedCamera(BaseMotionEstimator):
    """Every frame is the reference frame. Nothing is computed, which saves most of a moving camera's time."""

    def estimate(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> CameraMotion:
        return CameraMotion()
```

`src/tvd/motion/__init__.py`:

```python
"""Camera-motion estimators. Importing this package registers every estimator in it."""

from tvd.motion import fixed  # noqa: F401  Registers "fixed".
from tvd.motion.base import MOTION_ESTIMATORS, BaseMotionEstimator, CameraMotion

__all__ = ["MOTION_ESTIMATORS", "BaseMotionEstimator", "CameraMotion"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `67 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_motion.py src/tvd/motion/base.py src/tvd/motion/fixed.py src/tvd/motion/__init__.py
git commit -m "Add camera motion, and the fixed camera"
```

### Task 5: Moving camera: ORB

**Why:** The two test videos with signals and directions were filmed by a moving camera, so the notebooks' method comes over: ORB keypoints matched to the reference frame, with every detected object masked out. It is the slowest step of the pipeline, about 0.2 s per 1080p frame on the server's 4 CPU cores, which is why only a camera file that says so uses it. The test compares it with the notebooks on 40 frames: the transforms must be **equal**, not close.

**Files:**
- Replace: `tests/tvd/test_tvd_motion.py`
- Create: `src/tvd/motion/orb.py`
- Replace: `src/tvd/motion/__init__.py`

**Interfaces:**
- Consumes: `BaseMotionEstimator`, `CameraMotion`, `MOTION_ESTIMATORS` (Task 4).
- Produces: `OrbMotionEstimator(reference_frame, excluded_areas=(), features=6000, ratio_test=0.75, ransac_threshold=2.0, mask_margin=12, static_classes=("traffic light",), max_threads=32)` as `"orb"`, with `needs_detections = True` and `static_mask(shape, detections)`.

- [ ] **Step 1: Write the failing test**

`test_tvd_motion.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/test_tvd_motion.py`:

```python
import numpy as np
import pytest

from tvd.domain import Box, Detection
from tvd.motion import MOTION_ESTIMATORS, CameraMotion
from tvd.motion.orb import OrbMotionEstimator
from tvd.video import VideoReader


def detections_at(index: int) -> list[Detection]:
    return [Detection(Box(100 + index, 100, 200 + index, 180), 2, "car", 0.9), Detection(Box(300, 20, 320, 60), 9, "traffic light", 0.8)]


def test_the_estimators_are_registered():
    assert {"fixed", "orb"} <= set(MOTION_ESTIMATORS.names())


def test_a_fixed_camera_leaves_points_unchanged(sample_video):
    with VideoReader(sample_video) as reader:
        motion = MOTION_ESTIMATORS.create("fixed").estimate(reader, {}, 0, 10)
    assert motion.is_fixed and motion.to_reference(5, [[12.5, 7.25]]).tolist() == [[12.5, 7.25]]


def test_a_moving_camera_maps_both_ways():
    motion = CameraMotion(np.array([[1.0, 0.0, 10.0, -5.0], [0.0, 1.0, 0.0, 0.0]]), first_frame=100)
    assert motion.to_reference(100, [[0, 0]]).tolist() == [[10.0, -5.0]]
    assert motion.to_reference(101, [[2, 0]])[0].tolist() == pytest.approx([0.0, 2.0])  # A quarter turn.
    assert motion.to_frame(100, motion.to_reference(100, [[3, 4]]))[0].tolist() == pytest.approx([3.0, 4.0])


def test_a_frame_outside_the_estimate_is_refused():
    with pytest.raises(IndexError, match="Frame 102 is outside"):
        CameraMotion(np.zeros((2, 4)), first_frame=100).params_at(102)


def test_the_reference_frame_must_be_analysed(sample_video):
    with VideoReader(sample_video) as reader, pytest.raises(ValueError, match="Reference frame 50 is outside"):
        OrbMotionEstimator(reference_frame=50).estimate(reader, {}, 0, 10)


def test_the_mask_hides_moving_objects_and_drawn_areas_only():
    mask = OrbMotionEstimator(reference_frame=0, excluded_areas=[[0, 0, 5, 5]], mask_margin=2).static_mask((360, 640), detections_at(0))
    assert mask[0, 0] == 0 and mask[100, 98] == 0 and mask[100, 97] == 255  # Excluded area; a car's box plus its 2 px margin.
    assert mask[30, 310] == 255  # A traffic light is part of the scene.


def test_the_same_motion_as_the_notebooks(sample_video):
    from traffic_violation_detection.camera import estimate_camera_motion

    stop, reference = 40, 20
    notebook_detections = [[{"class_name": d.class_name, "bbox_xyxy": d.box.as_list()} for d in detections_at(k)] for k in range(stop)]
    expected = estimate_camera_motion(sample_video, notebook_detections, reference)
    with VideoReader(sample_video) as reader:
        motion = OrbMotionEstimator(reference_frame=reference).estimate(reader, {k: detections_at(k) for k in range(stop)}, 0, stop)
    assert np.array_equal(np.array([motion.params_at(k) for k in range(stop)]), expected)
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_motion.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.motion.orb'`

- [ ] **Step 3: Write the code**

Then replace `src/tvd/motion/__init__.py`, so that importing the package registers ORB too.

`src/tvd/motion/orb.py`:

```python
"""Camera motion from ORB keypoints matched to a reference frame, the method chosen in Notebook 04.

Keypoints on moving objects follow the traffic, not the camera, so every detected object is masked
out. Each frame is matched directly to the reference frame, which avoids accumulating drift.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import cv2
import numpy as np

from tvd.domain import Detection
from tvd.motion.base import IDENTITY, MOTION_ESTIMATORS, BaseMotionEstimator, CameraMotion
from tvd.video import VideoReader


@MOTION_ESTIMATORS.register("orb")
class OrbMotionEstimator(BaseMotionEstimator):
    """A moving camera. About 0.2 s per 1080p frame on 4 CPU cores, so use it only for a camera that moves.

    ``excluded_areas`` are ``[x1, y1, x2, y2]`` boxes of anything drawn over the image, such as a logo or
    a signal graphic, which does not move with the road. ``static_classes`` are detected classes that
    are part of the scene, so their keypoints stay usable.
    """

    needs_detections = True

    def __init__(
        self,
        reference_frame: int,
        excluded_areas: Sequence[Sequence[int]] = (),
        features: int = 6000,
        ratio_test: float = 0.75,
        ransac_threshold: float = 2.0,
        mask_margin: int = 12,
        static_classes: Sequence[str] = ("traffic light",),
        max_threads: int = 32,
    ) -> None:
        self.reference_frame = reference_frame
        self.excluded_areas = [tuple(int(v) for v in area) for area in excluded_areas]
        self.features = features
        self.ratio_test = ratio_test
        self.ransac_threshold = ransac_threshold
        self.mask_margin = mask_margin
        self.static_classes = set(static_classes)
        self.max_threads = max_threads

    def estimate(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> CameraMotion:
        if not start <= self.reference_frame < stop:
            raise ValueError(f"Reference frame {self.reference_frame} is outside the analysed frames {start} to {stop - 1}")
        # Importing Ultralytics leaves OpenCV on one thread. This step then takes about 4 times longer, with the same result.
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(min(self.max_threads, os.cpu_count() or 1))
        try:
            return CameraMotion(self._register(reader, detections, start, stop), first_frame=start)
        finally:
            cv2.setNumThreads(previous_threads)

    def static_mask(self, shape: tuple[int, int], detections: list[Detection]) -> np.ndarray:
        """255 where keypoints may be used: everywhere but the excluded areas and the moving objects."""
        mask = np.full(shape, 255, np.uint8)
        for x1, y1, x2, y2 in self.excluded_areas:
            mask[y1:y2, x1:x2] = 0
        margin = self.mask_margin
        for d in detections:
            if d.class_name in self.static_classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in d.box.as_list())
            mask[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin] = 0
        return mask

    def _fit(self, points, descriptors, reference_points, reference_descriptors, matcher) -> tuple[float, float, float, float] | None:
        """The transform from a frame to the reference, or None if the frame cannot be matched."""
        if descriptors is None or reference_descriptors is None:
            return None
        pairs = matcher.knnMatch(descriptors, reference_descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio_test * p[1].distance]
        if len(good) < 3:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(
            np.float32([points[m.queryIdx].pt for m in good]),
            np.float32([reference_points[m.trainIdx].pt for m in good]),
            method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=5000,
        )
        return None if matrix is None else (float(matrix[0, 0]), float(matrix[1, 0]), float(matrix[0, 2]), float(matrix[1, 2]))

    def _register(self, reader: VideoReader, detections: dict[int, list[Detection]], start: int, stop: int) -> np.ndarray:
        orb = cv2.ORB_create(self.features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        reference = next(reader.frames(self.reference_frame, self.reference_frame + 1))
        gray = cv2.cvtColor(reference.image, cv2.COLOR_BGR2GRAY)
        reference_points, reference_descriptors = orb.detectAndCompute(gray, self.static_mask(gray.shape, detections.get(self.reference_frame, [])))

        params = np.zeros((stop - start, 4))
        current = IDENTITY
        unmatched = 0
        for frame in reader.frames(start, stop):
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            points, descriptors = orb.detectAndCompute(gray, self.static_mask(gray.shape, detections.get(frame.index, [])))
            fitted = self._fit(points, descriptors, reference_points, reference_descriptors, matcher)
            unmatched += fitted is None
            current = fitted if fitted is not None else current
            params[frame.index - start] = current
        if unmatched:
            warnings.warn(f"Camera motion: {unmatched} of {len(params)} frames could not be matched and reuse the previous transform.", stacklevel=3)
        return params
```

`src/tvd/motion/__init__.py`:

```python
"""Camera-motion estimators. Importing this package registers every estimator in it."""

from tvd.motion import fixed, orb  # noqa: F401  Registers "fixed" and "orb".
from tvd.motion.base import MOTION_ESTIMATORS, BaseMotionEstimator, CameraMotion

__all__ = ["MOTION_ESTIMATORS", "BaseMotionEstimator", "CameraMotion"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `70 passed`.

`test_the_same_motion_as_the_notebooks` needs no GPU: it uses the committed sample video and made-up detections, which both implementations receive.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_motion.py src/tvd/motion/orb.py src/tvd/motion/__init__.py
git commit -m "Add the moving-camera estimator"
```

### Task 6: Signal readers

**Why:** A signal head is read differently by day (each lamp's colour) and by night (the glow around the head). Both are a `BaseSignalReader` with one method (**interface segregation**), registered as `lamps` and `glow`, so a camera file picks one by name. The notebooks call the night reader `head`; `glow` says what it looks at. The thresholds found in Notebook 05b are the defaults, and `min_pixels` can be changed per camera. A test reader that always shows one state joins `conftest.py`.

**Files:**
- Replace: `tests/tvd/conftest.py`
- Create: `tests/tvd/test_tvd_signals.py`
- Create: `src/tvd/signals/base.py`
- Create: `src/tvd/signals/lamps.py`
- Create: `src/tvd/signals/glow.py`
- Create: `src/tvd/signals/__init__.py`

**Interfaces:**
- Consumes: `Registry` (Part 1).
- Produces: `BaseSignalReader.read(image) -> str`; `SIGNAL_READERS`; `LampReader(red, yellow, green, half=6, min_pixels=6)` as `"lamps"`; `GlowReader(roi, min_pixels=12, min_mean_saturation=8)` as `"glow"`; `combine_states(states)`; `smooth_states(states, window=7)`; state names `RED`, `YELLOW`, `GREEN`, `UNKNOWN`, `CONFLICT`. Test double `test-fixed-state`.

- [ ] **Step 1: Write the failing tests**

`conftest.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/conftest.py`:

```python
"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame, TrackedObject
from tvd.signals import SIGNAL_READERS, BaseSignalReader
from tvd.tracking import TRACKERS, BaseTracker

ROOT = Path(__file__).resolve().parents[2]


@DETECTORS.register("test-moving-car")
class MovingCarDetector(BaseDetector):
    """A car that moves right by ``step`` px per frame, and a person who stands still."""

    def __init__(self, step: float = 5.0) -> None:
        self.step = step

    def detect(self, frame: Frame) -> list[Detection]:
        x = 10 + self.step * frame.index
        return [Detection(Box(x, 50, x + 40, 80), 2, "car", 0.9), Detection(Box(0, 0, 10, 30), 0, "person", 0.8)]


@TRACKERS.register("test-same-id")
class SameIdTracker(BaseTracker):
    """Gives every detection track id 1: enough to test what surrounds a tracker."""

    def __init__(self) -> None:
        self.resets = 0

    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        return [TrackedObject(1, d.box, d.class_id, d.class_name, d.confidence) for d in detections]

    def reset(self) -> None:
        self.resets += 1


@SIGNAL_READERS.register("test-fixed-state")
class FixedStateReader(BaseSignalReader):
    """A signal head that always shows ``state``."""

    def __init__(self, state: str = "red") -> None:
        self.state = state

    def read(self, image) -> str:
        return self.state


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
```

`tests/tvd/test_tvd_signals.py`:

```python
import numpy as np
import pytest

from tvd.signals import SIGNAL_READERS, combine_states, smooth_states

LAMPS = {"red": [20, 10], "yellow": [20, 30], "green": [20, 50]}
BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 255, 0)}


def image_with(colour: str | None, size: int = 6) -> np.ndarray:
    """A dark 64x64 image, with one lamp lit."""
    image = np.zeros((64, 64, 3), np.uint8)
    if colour:
        x, y = LAMPS[colour]
        image[y - size:y + size + 1, x - size:x + size + 1] = BGR[colour]
    return image


@pytest.mark.parametrize("colour", ["red", "yellow", "green"])
def test_lamps_read_the_lit_one(colour):
    assert SIGNAL_READERS.create("lamps", **LAMPS).read(image_with(colour)) == colour


def test_lamps_all_dark_are_unknown():
    assert SIGNAL_READERS.create("lamps", **LAMPS).read(image_with(None)) == "unknown"


def test_a_lamp_too_close_to_the_border_counts_nothing():
    assert SIGNAL_READERS.create("lamps", red=[2, 10], yellow=[20, 30], green=[20, 50]).read(image_with(None)) == "unknown"


def test_glow_of_a_grey_image_is_unknown():
    assert SIGNAL_READERS.create("glow", roi=[0, 0, 64, 64]).read(np.full((64, 64, 3), 200, np.uint8)) == "unknown"


def test_glow_reads_its_colour():
    assert SIGNAL_READERS.create("glow", roi=[0, 0, 64, 64]).read(image_with("green", size=10)) == "green"


@pytest.mark.parametrize("states, expected", [
    (["red", "red"], "red"), (["red", "unknown"], "red"), (["unknown", "unknown"], "unknown"), (["red", "green"], "conflict"), ([], "unknown"),
])
def test_heads_of_one_group_combine(states, expected):
    assert combine_states(states) == expected


def test_smoothing_removes_single_frame_flicker():
    assert smooth_states(["red"] * 4 + ["green"] + ["red"] * 4) == ["red"] * 9
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_signals.py -q
```

Expected: `ImportError while loading conftest ... ModuleNotFoundError: No module named 'tvd.signals'`

- [ ] **Step 3: Write the code**

`src/tvd/signals/base.py`:

```python
"""What every signal reader offers, and how the readings of several heads become one state."""

from abc import ABC, abstractmethod

import numpy as np

from tvd.registry import Registry

RED, YELLOW, GREEN = "red", "yellow", "green"
UNKNOWN = "unknown"    # Nothing could be read.
CONFLICT = "conflict"  # Heads of one group disagree.


class BaseSignalReader(ABC):
    """Reads the state of one signal head in one frame."""

    @abstractmethod
    def read(self, image: np.ndarray) -> str:
        """``red``, ``yellow``, ``green``, or ``unknown`` when the head cannot be read."""


SIGNAL_READERS: Registry[BaseSignalReader] = Registry("signal reader")


def combine_states(states: list[str]) -> str:
    """The state of a group of heads: ``unknown`` if none is readable, ``conflict`` if the readable ones disagree."""
    known = {s for s in states if s != UNKNOWN}
    if not known:
        return UNKNOWN
    return known.pop() if len(known) == 1 else CONFLICT


def smooth_states(states: list[str], window: int = 7) -> list[str]:
    """A majority vote over a sliding window, to remove single-frame flicker. A tie goes to the name sorted last."""
    half = window // 2
    smoothed = []
    for i in range(len(states)):
        votes = states[max(0, i - half):i + half + 1]
        smoothed.append(max(set(votes), key=lambda s: (votes.count(s), s)))
    return smoothed
```

`src/tvd/signals/lamps.py`:

```python
"""Day: the colour at each lamp's position."""

import cv2
import numpy as np

from tvd.signals.base import GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader


@SIGNAL_READERS.register("lamps")
class LampReader(BaseSignalReader):
    """Counts lit pixels in a small square around each lamp of one head.

    Each lamp is read separately, because a head's yellow backplate confuses a count over the whole
    head. A colour wins if it has at least ``min_pixels`` and twice as many as the others together.
    """

    def __init__(self, red: list[int], yellow: list[int], green: list[int], half: int = 6, min_pixels: int = 6) -> None:
        self.lamps = {RED: tuple(red), YELLOW: tuple(yellow), GREEN: tuple(green)}
        self.half = half
        self.min_pixels = min_pixels

    def read(self, image: np.ndarray) -> str:
        counts = {}
        height, width = image.shape[:2]
        half = self.half
        for colour, (x, y) in self.lamps.items():
            if not (half <= x < width - half and half <= y < height - half):
                counts[colour] = 0
                continue
            hsv = cv2.cvtColor(image[y - half:y + half + 1, x - half:x + half + 1], cv2.COLOR_BGR2HSV)
            counts[colour] = int(lit_pixels(colour, hsv[..., 0], hsv[..., 1], hsv[..., 2]).sum())
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels and counts[best] >= 2 * (sum(counts.values()) - counts[best]) else UNKNOWN


def lit_pixels(colour: str, hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Pixels lit in ``colour``, with the HSV thresholds found in Notebook 05b. OpenCV's hue runs from 0 to 179."""
    if colour == RED:
        return ((hue <= 10) | (hue >= 165)) & (saturation >= 90) & (value >= 90)
    if colour == YELLOW:
        return (hue >= 14) & (hue <= 35) & (saturation >= 120) & (value >= 170)
    return (hue >= 40) & (hue <= 105) & (saturation >= 60) & (value >= 110)
```

`src/tvd/signals/glow.py`:

```python
"""Night: the colour of the glow around a whole head."""

import cv2
import numpy as np

from tvd.signals.base import GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader


@SIGNAL_READERS.register("glow")
class GlowReader(BaseSignalReader):
    """Counts lit pixels of each colour in a box around one head (``roi``, as ``[x1, y1, x2, y2]``).

    A monochrome (infrared) image has no colour to read, and gives ``unknown``. A colour wins if it has
    at least ``min_pixels`` and twice as many as the others together.
    """

    def __init__(self, roi: list[int], min_pixels: int = 12, min_mean_saturation: float = 8) -> None:
        self.roi = tuple(int(v) for v in roi)
        self.min_pixels = min_pixels
        self.min_mean_saturation = min_mean_saturation

    def read(self, image: np.ndarray) -> str:
        x1, y1, x2, y2 = self.roi
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if saturation.mean() < self.min_mean_saturation:
            return UNKNOWN
        lit = (value >= 120) & (saturation >= 70)
        counts = {
            RED: int((lit & ((hue <= 10) | (hue >= 165))).sum()),
            YELLOW: int((lit & (hue >= 14) & (hue <= 35)).sum()),
            GREEN: int((lit & (hue >= 40) & (hue <= 105)).sum()),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels and counts[best] >= 2 * (sum(counts.values()) - counts[best]) else UNKNOWN
```

`src/tvd/signals/__init__.py`:

```python
"""Signal readers. Importing this package registers every reader in it."""

from tvd.signals import glow, lamps  # noqa: F401  Registers "glow" and "lamps".
from tvd.signals.base import CONFLICT, GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader, combine_states, smooth_states

__all__ = ["CONFLICT", "GREEN", "RED", "SIGNAL_READERS", "UNKNOWN", "YELLOW", "BaseSignalReader", "combine_states", "smooth_states"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `83 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/conftest.py tests/tvd/test_tvd_signals.py src/tvd/signals/base.py src/tvd/signals/lamps.py src/tvd/signals/glow.py src/tvd/signals/__init__.py
git commit -m "Add the day and night signal readers"
```

### Task 7: Signal states over a video

**Why:** `SignalMonitor` is a `FrameObserver`: during the pass it reads every group of heads in every frame, and at the end smooths each group's states into a `SignalTimeline`, which the red-light rule asks for the state of a group at a frame. `notebook_data.py` holds the helpers that turn the notebooks' scene files and saved results into what `tvd` takes; this and later tasks compare with the notebooks through it. On the server, the comparison reads 90 frames of three clips: day lamps, night glow, and infrared, where nothing can be read.

**Files:**
- Create: `tests/tvd/notebook_data.py`
- Replace: `tests/tvd/test_tvd_signals.py`
- Create: `src/tvd/signals/monitor.py`
- Replace: `src/tvd/signals/__init__.py`

**Interfaces:**
- Consumes: `FrameObserver` (Task 2), the readers and `combine_states`, `smooth_states` (Task 6).
- Produces: `SignalMonitor(groups: dict[str, list[BaseSignalReader]], smoothing_frames=7)` with `observe` and `timeline()`; `SignalTimeline(states)` with `get(group, frame, default=None)`, `state(group, frame)` (`"unknown"` outside), `groups`, `as_dict()`. Test helpers in `tests/tvd/notebook_data.py`.

- [ ] **Step 1: Write the failing tests**

`test_tvd_signals.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/notebook_data.py`:

```python
"""The notebooks' scene files and saved results, turned into what the tvd package takes, for tests that compare the two.

The saved results are in outputs/, which only the GPU server has: tests that need them skip elsewhere.
"""

import json
from pathlib import Path

from tvd.domain import Box, TrackedObject
from tvd.tracking.history import TrackHistory

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
RED_LIGHT_SCENE = ROOT / "configs" / "scenes" / "rlr_occasions.json"
RLR_VIDEO = ROOT / "data" / "raw" / "rlr_occasions.mp4"
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def wrong_way_scene(scene: dict) -> dict:
    """A notebook wrong-way scene as the ``scene`` of a camera file's ``wrong_way`` rule: only regions with a direction matter."""
    return {
        "vanishing_point": scene["vanishing_point"],
        "regions": [
            {"id": r["id"], "polygon": r["polygon"], "allowed_direction": r["allowed_direction"], "tolerance_deg": r["direction_tolerance_deg"]}
            for r in scene["regions"] if r["allowed_direction"]
        ],
    }


def red_light_section(scene: dict, clip_name: str) -> dict:
    """One clip of a notebook red-light scene as the ``red_light`` section of a camera file.

    The notebooks call the night reader "head"; here it is "glow", and each head is ``{"roi": [x1, y1, x2, y2]}``.
    """
    clip = scene["clips"][clip_name]
    signals = clip["signals"]
    lamps = signals["method"] == "lamps"
    options = {"half": signals["lamp_half"]} if lamps and "lamp_half" in signals else {}
    head = (lambda h: dict(h)) if lamps else (lambda h: {"roi": list(h)})
    return {
        "scene": {
            "stop_line": clip["stop_line"],
            "lane_dividers": clip["lane_dividers"],
            "lane_direction_deg": scene["lane_direction_deg"],
            "traffic_direction": clip.get("traffic_direction", "away_from_camera"),
            "evaluable_x_max": clip.get("evaluable_x_max"),
            "signal_source": scene.get("signal_source", "camera"),
            "signals": {
                "reader": "lamps" if lamps else "glow",
                "options": options,
                "through": [head(h) for h in signals["through"]],
                "left": [head(h) for h in signals["left"]],
            },
        },
        "settings": {k: clip[k] for k in ("entry_offset", "pass_margin") if k in clip},
    }


def history_from_rows(tracks: dict[str, list]) -> TrackHistory:
    """A ``TrackHistory`` from the notebooks' rows of ``frame, x1, y1, x2, y2, class_id, confidence``, in the same order of tracks."""
    by_frame: dict[int, list[TrackedObject]] = {}
    for track_id, rows in tracks.items():
        for frame, x1, y1, x2, y2, class_id, confidence in rows:
            by_frame.setdefault(int(frame), []).append(TrackedObject(int(track_id), Box(x1, y1, x2, y2), int(class_id), VEHICLE_CLASSES[int(class_id)], confidence))
    history = TrackHistory()
    for frame in sorted(by_frame):
        history.add(frame, by_frame[frame])
    return history


def load_json(path: Path):
    return json.loads(path.read_text())
```

`tests/tvd/test_tvd_signals.py`:

```python
import numpy as np
import pytest
from notebook_data import RED_LIGHT_SCENE, RLR_VIDEO, load_json, red_light_section

from tvd.domain import Frame
from tvd.signals import SIGNAL_READERS, SignalMonitor, combine_states, smooth_states
from tvd.video import VideoReader

needs_rlr_video = pytest.mark.skipif(not RLR_VIDEO.is_file(), reason="Needs data/raw/rlr_occasions.mp4, which only the GPU server has.")
LAMPS = {"red": [20, 10], "yellow": [20, 30], "green": [20, 50]}
BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 255, 0)}


def image_with(colour: str | None, size: int = 6) -> np.ndarray:
    """A dark 64x64 image, with one lamp lit."""
    image = np.zeros((64, 64, 3), np.uint8)
    if colour:
        x, y = LAMPS[colour]
        image[y - size:y + size + 1, x - size:x + size + 1] = BGR[colour]
    return image


@pytest.mark.parametrize("colour", ["red", "yellow", "green"])
def test_lamps_read_the_lit_one(colour):
    assert SIGNAL_READERS.create("lamps", **LAMPS).read(image_with(colour)) == colour


def test_lamps_all_dark_are_unknown():
    assert SIGNAL_READERS.create("lamps", **LAMPS).read(image_with(None)) == "unknown"


def test_a_lamp_too_close_to_the_border_counts_nothing():
    assert SIGNAL_READERS.create("lamps", red=[2, 10], yellow=[20, 30], green=[20, 50]).read(image_with(None)) == "unknown"


def test_glow_of_a_grey_image_is_unknown():
    assert SIGNAL_READERS.create("glow", roi=[0, 0, 64, 64]).read(np.full((64, 64, 3), 200, np.uint8)) == "unknown"


def test_glow_reads_its_colour():
    assert SIGNAL_READERS.create("glow", roi=[0, 0, 64, 64]).read(image_with("green", size=10)) == "green"


@pytest.mark.parametrize("states, expected", [
    (["red", "red"], "red"), (["red", "unknown"], "red"), (["unknown", "unknown"], "unknown"), (["red", "green"], "conflict"), ([], "unknown"),
])
def test_heads_of_one_group_combine(states, expected):
    assert combine_states(states) == expected


def test_smoothing_removes_single_frame_flicker():
    assert smooth_states(["red"] * 4 + ["green"] + ["red"] * 4) == ["red"] * 9


def test_a_monitor_smooths_each_group_and_leaves_a_group_without_heads_unknown():
    monitor = SignalMonitor({"through": [SIGNAL_READERS.create("test-fixed-state", state="green")], "left": []}, smoothing_frames=3)
    for index in range(5, 8):
        monitor.observe(Frame(index, 0.0, np.zeros((2, 2, 3), np.uint8)), [])
    timeline = monitor.timeline()
    assert timeline.as_dict() == {"through": {5: "green", 6: "green", 7: "green"}, "left": {5: "unknown", 6: "unknown", 7: "unknown"}}
    assert timeline.get("through", 8) is None and timeline.state("through", 8) == "unknown"


@needs_rlr_video
@pytest.mark.parametrize("clip_name", ["A", "C", "H"])  # Day lamps, night glow, infrared.
def test_the_same_states_as_the_notebooks(clip_name):
    from traffic_violation_detection.red_light import read_frame_signals, smooth_states as notebook_smooth

    scene = load_json(RED_LIGHT_SCENE)
    clip = scene["clips"][clip_name]
    signals = red_light_section(scene, clip_name)["scene"]["signals"]
    groups = {g: [SIGNAL_READERS.create(signals["reader"], **signals["options"], **head) for head in signals[g]] for g in ("through", "left")}
    monitor = SignalMonitor(groups)
    raw = {"through": [], "left": []}
    with VideoReader(RLR_VIDEO) as reader:
        for frame in reader.frames(clip["start"], clip["start"] + 90):
            monitor.observe(frame, [])
            for group, state in read_frame_signals(frame.image, clip).items():
                raw[group].append(state)
    expected = {group: {clip["start"] + i: s for i, s in enumerate(notebook_smooth(states))} for group, states in raw.items()}
    assert monitor.timeline().as_dict() == expected
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_signals.py -q
```

Expected: `E   ImportError: cannot import name 'SignalMonitor' from 'tvd.signals'`

- [ ] **Step 3: Write the code**

Then replace `src/tvd/signals/__init__.py`, so that the package exports the monitor and the timeline.

`src/tvd/signals/monitor.py`:

```python
"""Signal states over a video: read in every frame of the pass, then smoothed."""

from __future__ import annotations

from tvd.domain import Detection, Frame
from tvd.observers import FrameObserver
from tvd.signals.base import UNKNOWN, BaseSignalReader, combine_states, smooth_states


class SignalTimeline:
    """The smoothed state of each signal group in each analysed frame."""

    def __init__(self, states: dict[str, dict[int, str]]) -> None:
        self._states = states

    @property
    def groups(self) -> list[str]:
        return list(self._states)

    def get(self, group: str, frame_index: int, default: str | None = None) -> str | None:
        """The state of a group in a frame, or ``default`` for a frame that was not analysed."""
        return self._states.get(group, {}).get(frame_index, default)

    def state(self, group: str, frame_index: int) -> str:
        return self.get(group, frame_index, UNKNOWN)

    def as_dict(self) -> dict[str, dict[int, str]]:
        return {group: dict(states) for group, states in self._states.items()}


class SignalMonitor(FrameObserver):
    """Reads every group of signal heads in every frame. A group without heads is always ``unknown``.

    The states are smoothed only at the end, by ``timeline()``, because a sliding window needs the frames after each one.
    """

    def __init__(self, groups: dict[str, list[BaseSignalReader]], smoothing_frames: int = 7) -> None:
        self.groups = groups
        self.smoothing_frames = smoothing_frames
        self._frames: list[int] = []
        self._raw: dict[str, list[str]] = {group: [] for group in groups}

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self._frames.append(frame.index)
        for group, readers in self.groups.items():
            self._raw[group].append(combine_states([r.read(frame.image) for r in readers]) if readers else UNKNOWN)

    def timeline(self) -> SignalTimeline:
        return SignalTimeline({
            group: dict(zip(self._frames, smooth_states(raw, self.smoothing_frames))) for group, raw in self._raw.items()
        })
```

`src/tvd/signals/__init__.py`:

```python
"""Signal readers. Importing this package registers every reader in it."""

from tvd.signals import glow, lamps  # noqa: F401  Registers "glow" and "lamps".
from tvd.signals.base import CONFLICT, GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader, combine_states, smooth_states
from tvd.signals.monitor import SignalMonitor, SignalTimeline

__all__ = [
    "CONFLICT", "GREEN", "RED", "SIGNAL_READERS", "UNKNOWN", "YELLOW",
    "BaseSignalReader", "SignalMonitor", "SignalTimeline", "combine_states", "smooth_states",
]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `87 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/notebook_data.py tests/tvd/test_tvd_signals.py src/tvd/signals/monitor.py src/tvd/signals/__init__.py
git commit -m "Read signal states over a video"
```

### Task 8: Rule interface

**Why:** `BaseRule` is the contract of every violation rule: it gets an `Observation` (the tracks, the camera's motion, the video's size and frame rate) and returns a `RuleResult`: findings for review, one record per judged track, and the settings it used. A rule that must see frames, as the red-light rule does for its signals, returns its observers, and the pass feeds them without knowing what they are. `config.validate` checks one section of a file, so that a rule reports a problem as `settings.window_seconds: ...`.

**Files:**
- Replace: `tests/tvd/test_tvd_config.py`
- Create: `tests/tvd/test_tvd_rules.py`
- Replace: `src/tvd/config.py`
- Create: `src/tvd/rules/base.py`
- Create: `src/tvd/rules/__init__.py`

**Interfaces:**
- Consumes: `CameraMotion` (Task 4), `FrameObserver` (Task 2), `Track` (Task 1), `Registry`, `StrictModel` (Part 1).
- Produces: `Observation(tracks, motion, fps, width, height, first_frame, end_frame)`; `Finding(rule, track_id, class_name, start_frame, trigger_frame, end_frame, why, evidence)`, frozen; `RuleResult(findings, records, settings)` with `verdict_counts()`; `BaseRule` with `name`, `frame_observers()` and abstract `evaluate(observation)`; `RULES`; `describe(error, where)` and `validate(model, data, where="") -> model` in `config.py`.

- [ ] **Step 1: Write the failing tests**

`test_tvd_config.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/test_tvd_config.py`:

```python
import pytest
from conftest import ROOT
from pydantic import ValidationError

from tvd.config import ConfigError, PerceptionConfig, load_config, validate

GOOD = """
detector:
  type: test-moving-car
  options: {step: 2}
tracker:
  type: test-same-id
track_classes: [car]
"""


def write(tmp_path, text: str):
    path = tmp_path / "perception.yaml"
    path.write_text(text)
    return path


def test_a_good_file_loads(tmp_path):
    config = load_config(write(tmp_path, GOOD), PerceptionConfig)
    assert config.detector.options == {"step": 2} and config.tracker.options == {} and config.track_classes == ["car"]


def test_options_left_empty_mean_no_options(tmp_path):
    config = load_config(write(tmp_path, GOOD.replace("type: test-same-id", "type: test-same-id\n  options:")), PerceptionConfig)
    assert config.tracker.options == {}


def test_the_committed_settings_load():
    config = load_config(ROOT / "configs" / "tvd" / "perception.yaml", PerceptionConfig)
    assert (config.detector.type, config.tracker.type, config.tracker.options) == ("yolo", "ocsort", {"use_byte": True})


def test_an_unknown_key_is_named(tmp_path):
    with pytest.raises(ConfigError, match="trackr: Extra inputs are not permitted"):
        load_config(write(tmp_path, GOOD + "trackr: ocsort\n"), PerceptionConfig)


def test_a_missing_key_is_named(tmp_path):
    with pytest.raises(ConfigError, match="tracker: Field required"):
        load_config(write(tmp_path, GOOD.replace("tracker:\n  type: test-same-id\n", "")), PerceptionConfig)


def test_a_wrong_type_is_named(tmp_path):
    with pytest.raises(ConfigError, match="track_classes: Input should be a valid list"):
        load_config(write(tmp_path, GOOD.replace("[car]", "car")), PerceptionConfig)


def test_an_empty_list_of_classes_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="track_classes: List should have at least 1 item"):
        load_config(write(tmp_path, GOOD.replace("[car]", "[]")), PerceptionConfig)


def test_every_problem_is_reported_at_once(tmp_path):
    text = GOOD.replace("tracker:\n  type: test-same-id\n", "") + "trackr: ocsort\n"
    with pytest.raises(ConfigError, match="tracker: Field required; trackr: Extra inputs are not permitted"):
        load_config(write(tmp_path, text), PerceptionConfig)


def test_a_file_that_is_not_yaml_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "detector: [unclosed"), PerceptionConfig)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml", PerceptionConfig)


def test_settings_cannot_change_after_loading(tmp_path):
    config = load_config(write(tmp_path, GOOD), PerceptionConfig)
    with pytest.raises(ValidationError):
        config.track_classes = ["truck"]


def test_a_section_is_checked_with_its_place_in_the_file():
    with pytest.raises(ConfigError, match="rules.red_light.tracker: Field required"):
        validate(PerceptionConfig, {"detector": {"type": "x"}, "track_classes": ["car"]}, "rules.red_light")
```

`tests/tvd/test_tvd_rules.py`:

```python
import pytest

from tvd.motion import CameraMotion
from tvd.rules import BaseRule, Finding, Observation, RuleResult


def test_a_rule_must_implement_evaluate():
    class Incomplete(BaseRule):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_a_rule_needs_no_frames_unless_it_says_so():
    class Counting(BaseRule):
        name = "counting"

        def evaluate(self, observation):
            return RuleResult([], [{"track_id": str(t), "verdict": "seen"} for t in observation.tracks], {})

    rule = Counting()
    result = rule.evaluate(Observation({1: None, 2: None}, CameraMotion(), 30.0, 640, 360, 0, 10))
    assert rule.frame_observers() == [] and result.verdict_counts() == {"seen": 2}


def test_a_finding_cannot_be_changed():
    finding = Finding("wrong_way", "7", "car", 10, 20, 30, "why")
    with pytest.raises(AttributeError):
        finding.trigger_frame = 25
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_config.py tests/tvd/test_tvd_rules.py -q
```

Expected: `2 errors`, one for each file: `E   ImportError: cannot import name 'validate' from 'tvd.config'` and `E   ModuleNotFoundError: No module named 'tvd.rules'`

- [ ] **Step 3: Write the code**

`src/tvd/config.py`:

```python
"""Settings files in YAML, checked against models, so that a wrong key or value is reported with its file and its place."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(ValueError):
    """A settings file that cannot be used, with every problem found in it."""


class StrictModel(BaseModel):
    """Refuses keys it does not know, so that a misspelt setting is never silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentConfig(StrictModel):
    """Which implementation to build, by its registered name, and the options given to it."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def no_options_when_left_empty(cls, value: Any) -> Any:
        """``options:`` with nothing under it reads as null in YAML, and means no options."""
        return {} if value is None else value


class PerceptionConfig(StrictModel):
    """How to detect and track."""

    detector: ComponentConfig
    tracker: ComponentConfig
    track_classes: list[str] = Field(min_length=1)


M = TypeVar("M", bound=BaseModel)


def describe(error: ValidationError, where: str = "") -> str:
    """Every problem found by pydantic, as place: message, joined by ; . where is put before each place."""

    def place(location: tuple) -> str:
        return ".".join([where] * bool(where) + [str(part) for part in location]) or "(file)"

    return "; ".join(f"{place(e['loc'])}: {e['msg']}" for e in error.errors())


def validate(model: type[M], data: Any, where: str = "") -> M:
    """Check data already read, such as one section of a file, against ``model``. Problems raise a ``ConfigError``."""
    try:
        return model.model_validate(data if data is not None else {})
    except ValidationError as error:
        raise ConfigError(describe(error, where)) from error


def load_config(path: str | Path, model: type[M]) -> M:
    """Read a YAML file and check it against ``model``. Every problem is reported at once, in a ``ConfigError``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Settings file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: not valid YAML: {error}") from error
    try:
        return validate(model, data)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error
```

`src/tvd/rules/base.py`:

```python
"""What every violation rule offers, and what it is given and gives back."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tvd.motion import CameraMotion
from tvd.observers import FrameObserver
from tvd.registry import Registry
from tvd.tracking.history import Track


@dataclass(frozen=True)
class Observation:
    """What one pass over a video saw: what every rule judges. ``end_frame`` is one past the last frame analysed."""

    tracks: dict[int, Track]
    motion: CameraMotion
    fps: float
    width: int
    height: int
    first_frame: int
    end_frame: int


@dataclass(frozen=True)
class Finding:
    """A candidate violation, for a person to review. A rule never decides that a violation happened.

    ``start_frame`` to ``end_frame`` is the stretch of video that shows it; ``trigger_frame`` is the moment the rule fired.
    """

    rule: str
    track_id: str
    class_name: str
    start_frame: int
    trigger_frame: int
    end_frame: int
    why: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    """A rule's findings, a record of how it judged each track, and the settings it used."""

    findings: list[Finding]
    records: list[dict[str, Any]]
    settings: dict[str, Any]

    def verdict_counts(self) -> dict[str, int]:
        return dict(Counter(r["verdict"] for r in self.records))


class BaseRule(ABC):
    """Judges the tracks of one pass over a video.

    A rule that needs to see every frame, as the red-light rule reads the signals, returns its observers
    from ``frame_observers``, and the pass feeds them. The pass never needs to know which rules exist.
    """

    name: str

    def frame_observers(self) -> list[FrameObserver]:
        return []

    @abstractmethod
    def evaluate(self, observation: Observation) -> RuleResult:
        """Judge every track of the observation."""


RULES: Registry[BaseRule] = Registry("rule")
```

`src/tvd/rules/__init__.py`:

```python
"""Violation rules. Importing this package registers every rule in it."""

from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult

__all__ = ["RULES", "BaseRule", "Finding", "Observation", "RuleResult"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `91 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_config.py tests/tvd/test_tvd_rules.py src/tvd/config.py src/tvd/rules/base.py src/tvd/rules/__init__.py
git commit -m "Add the rule interface"
```

### Task 9: Wrong-way rule

**Why:** The wrong-way rule of Notebook 05, rewritten as a `BaseRule`. Its scene (vanishing point, directional regions) and its settings are two pydantic models, and **every setting comes from `configs/tvd/rules.yaml`**: the models have no defaults, so nothing that can be tuned is written in code. The notebook test replays the tracks and camera motion that Notebooks 03 and 04 saved on the server, forwards and then backwards, where vehicles go the wrong way, and requires the same verdict for every track and the same candidates.

**Files:**
- Create: `tests/tvd/test_tvd_wrong_way.py`
- Create: `src/tvd/rules/wrong_way.py`
- Replace: `src/tvd/rules/__init__.py`
- Create: `configs/tvd/rules.yaml`

**Interfaces:**
- Consumes: `BaseRule`, `Observation`, `Finding`, `RuleResult`, `RULES`, `validate` (Task 8); `point_in_polygon` (Task 3); `CameraMotion.to_reference` (Task 4).
- Produces: `WrongWayRule(scene: dict, settings: dict)` as `"wrong_way"`, with `scene`, `settings`, `measure(track, observation) -> TrackPath`, `windows(path, durations)`, `usable(window)`, `failed_conditions(window)`; `Region`, `WrongWayScene`, `WrongWaySettings`, `Durations.at(settings, fps)`; verdicts `candidate`, `no_violation`, `not_evaluable`, `not_in_directional_region`. `configs/tvd/rules.yaml` with its `wrong_way` section.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_wrong_way.py`:

```python
import numpy as np
import pytest
import yaml
from conftest import ROOT
from notebook_data import OUTPUTS, history_from_rows, load_json, wrong_way_scene

from tvd.config import ConfigError
from tvd.domain import Box, TrackedObject
from tvd.motion import CameraMotion
from tvd.rules import RULES, Observation
from tvd.rules.wrong_way import WrongWayRule
from tvd.tracking.history import TrackHistory

SETTINGS = yaml.safe_load((ROOT / "configs" / "tvd" / "rules.yaml").read_text())["wrong_way"]
# One region over the whole 640x360 image, where traffic may only go up the image (towards a far vanishing point).
SCENE = {"vanishing_point": [320, -100000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360], [0, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}
STORED_TRACKS = OUTPUTS / "03_multi_object_tracking" / "tracks" / "video-test.ocsort-byte.jsonl"
STORED_MOTION = OUTPUTS / "04_traffic_scene_configuration" / "video-test.camera_motion.json"
needs_stored_tracks = pytest.mark.skipif(not (STORED_TRACKS.is_file() and STORED_MOTION.is_file()), reason="Needs the outputs of Notebooks 03 and 04, as on the GPU server.")


def judge(frames: int, dy: float, top_dy: float | None = None, scene: dict = SCENE, settings: dict = SETTINGS):
    """One car of 40x30 px, moving down the image by ``dy`` px per frame (its top edge by ``top_dy``), at 24 FPS.

    Going up, it starts low in the image, so that it stays away from the border.
    """
    history = TrackHistory()
    start = 130 if dy >= 0 else 330
    for k in range(frames):
        y2 = start + dy * k
        y1 = start - 30 + (dy if top_dy is None else top_dy) * k
        history.add(k, [TrackedObject(7, Box(300, y1, 340, y2), 2, "car", 0.9)])
    return WrongWayRule(scene, settings).evaluate(Observation(history.tracks, CameraMotion(), 24.0, 640, 360, 0, frames))


def test_the_rule_is_registered():
    assert "wrong_way" in RULES.names()


def test_moving_against_the_direction_for_the_whole_window_is_a_candidate():
    result = judge(60, dy=3)
    (finding,) = result.findings
    assert result.records == [{"track_id": "7", "verdict": "candidate", "evaluable_windows": 12}]
    assert (finding.start_frame, finding.trigger_frame, finding.end_frame) == (0, 48, 59) and finding.evidence["region_id"] == "road"
    assert finding.why == "Moved against the allowed direction of road by 4.80 box heights in about 2.0 s, 0 degrees from the opposite direction."


def test_moving_with_the_direction_is_no_violation():
    assert judge(60, dy=-3).verdict_counts() == {"no_violation": 1}


def test_a_track_shorter_than_the_window_cannot_be_judged():
    assert judge(30, dy=3).verdict_counts() == {"not_evaluable": 1}


def test_a_box_that_only_grows_is_no_violation():
    assert judge(60, dy=3, top_dy=0).verdict_counts() == {"no_violation": 1}  # C4: the top edge stays still.


def test_a_track_outside_every_region_is_not_judged():
    elsewhere = {**SCENE, "regions": [{**SCENE["regions"][0], "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]}]}
    assert judge(60, dy=3, scene=elsewhere).verdict_counts() == {"not_in_directional_region": 1}


def test_the_window_comes_from_the_settings():
    # 60 frames at 24 FPS are 2.5 s: enough for a 2 s window, not for a 3 s one.
    assert judge(60, dy=3, settings={**SETTINGS, "window_seconds": 3.0}).verdict_counts() == {"not_evaluable": 1}


def test_a_wrong_setting_is_named():
    with pytest.raises(ConfigError, match="settings.window_seconds: Input should be greater than 0"):
        WrongWayRule(SCENE, {**SETTINGS, "window_seconds": 0})


@needs_stored_tracks
@pytest.mark.parametrize("reverse", [False, True])  # Played backwards, every vehicle goes the wrong way.
def test_the_same_results_as_the_notebooks(reverse):
    import json

    from traffic_violation_detection import wrong_way as notebook
    from traffic_violation_detection.scene import load_camera_motion
    from traffic_violation_detection.tracking import add_to_tracks
    from traffic_violation_detection.video import read_video_metadata

    tracks: dict[str, list] = {}
    for line in STORED_TRACKS.read_text().splitlines():
        record = json.loads(line)
        add_to_tracks(tracks, record["frame_index"], record["tracks"])
    params = load_camera_motion(STORED_MOTION)
    if reverse:
        last = len(params) - 1
        tracks = {t: [[last - r[0], *r[1:]] for r in reversed(rows)] for t, rows in tracks.items()}
        params = params[::-1].copy()
    fps = read_video_metadata(ROOT / "data" / "video-test.mp4")["fps"]
    scene = load_json(ROOT / "configs" / "scenes" / "video-test.json")
    expected = {t: notebook.evaluate_track(notebook.build_track(t, rows, params, 1920, 1080), scene, notebook.rule_for_fps(fps), "video-test", fps) for t, rows in tracks.items()}

    rule = WrongWayRule(wrong_way_scene(scene), SETTINGS)
    result = rule.evaluate(Observation(history_from_rows(tracks).tracks, CameraMotion(params), fps, 1920, 1080, 0, len(params)))

    assert {r["track_id"]: (r["verdict"], r["evaluable_windows"]) for r in result.records} == {t: (e["status"], e["evaluable_windows"]) for t, e in expected.items()}
    events = sorted((e for r in expected.values() for e in r["events"]), key=lambda e: (e["trigger_frame"], e["track_id"]))
    assert len(events) > 0 or not reverse
    assert sorted(([f.track_id, f.class_name, f.start_frame, f.trigger_frame, f.end_frame, f.why, f.evidence] for f in result.findings), key=lambda f: (f[3], f[0])) == [
        [e["track_id"], e["class_name"], e["start_frame"], e["trigger_frame"], e["end_frame"], e["why"], {"region_id": e["region_id"], **e["evidence"]}] for e in events
    ]
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_wrong_way.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.rules.wrong_way'`

- [ ] **Step 3: Write the code**

Replace `src/tvd/rules/__init__.py`, so that importing the package registers the rule, and create the settings file. Only the wrong-way section exists yet; Task 11 adds the red-light one.

`src/tvd/rules/wrong_way.py`:

```python
"""Wrong-way rule, developed in Notebook 05. Its output is a candidate for review, never a verdict.

A track gives a candidate if, in some window (2 s by default), all of these hold:

- C1: the window's midpoint lies in a region with an allowed direction;
- C2: its net movement points against that direction, within the region's tolerance;
- C3: the opposing distance is at least ``min_opposing_heights`` box heights;
- C4: the top edge of the box also moves against the direction, so a box that only grows does not count;
- C5: enough of the window is observed, and no box touches the image border.

Positions are measured in reference pixels, where the regions are drawn, so a moving camera is allowed for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy.ndimage import median_filter

from tvd.config import StrictModel, validate
from tvd.geometry import point_in_polygon
from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult
from tvd.tracking.history import Track

CANDIDATE = "candidate"
NO_VIOLATION = "no_violation"
NOT_EVALUABLE = "not_evaluable"
NOT_IN_REGION = "not_in_directional_region"


class Region(StrictModel):
    """A part of the road where traffic has one allowed direction, drawn on the reference frame."""

    id: str
    polygon: list[tuple[float, float]] = Field(min_length=3)
    allowed_direction: Literal["away_from_camera", "toward_camera"]
    tolerance_deg: float = Field(gt=0, le=180)


class WrongWayScene(StrictModel):
    """Lanes converge in the image, so a direction is given relative to the vanishing point: away from the camera points at it."""

    vanishing_point: tuple[float, float]
    regions: list[Region] = Field(min_length=1)


class WrongWaySettings(StrictModel):
    version: int
    window_seconds: float = Field(gt=0)
    smoothing_seconds: float = Field(ge=0)
    merge_gap_seconds: float = Field(ge=0)
    min_opposing_heights: float = Field(gt=0)
    min_top_edge_share: float = Field(ge=0, le=1)
    min_observed_fraction: float = Field(gt=0, le=1)
    border_margin: int = Field(ge=0)


@dataclass(frozen=True)
class Durations:
    """The rule's durations in frames, for one video's frame rate."""

    window: int
    smoothing: int
    merge_gap: int

    @classmethod
    def at(cls, settings: WrongWaySettings, fps: float) -> Durations:
        # A median filter needs an odd size.
        return cls(round(settings.window_seconds * fps), round(settings.smoothing_seconds * fps) | 1, round(settings.merge_gap_seconds * fps))


@dataclass(frozen=True)
class TrackPath:
    """One vehicle's measurements: its bottom and top centres in reference pixels, its height, and border contact."""

    track_id: str
    class_name: str
    frames: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    heights: np.ndarray
    clipped: np.ndarray


def allowed_direction(region: Region, position: np.ndarray, vanishing_point: tuple[float, float]) -> np.ndarray:
    """Unit vector of the direction a vehicle may travel at a position of a region."""
    position, vanishing = np.asarray(position, dtype=np.float64), np.asarray(vanishing_point, dtype=np.float64)
    unit = (vanishing - position) / np.linalg.norm(vanishing - position)
    return unit if region.allowed_direction == "away_from_camera" else -unit


@RULES.register("wrong_way")
class WrongWayRule(BaseRule):
    name = "wrong_way"

    def __init__(self, scene: dict[str, Any], settings: dict[str, Any]) -> None:
        self.scene = validate(WrongWayScene, scene, "scene")
        self.settings = validate(WrongWaySettings, settings, "settings")

    def evaluate(self, observation: Observation) -> RuleResult:
        durations = Durations.at(self.settings, observation.fps)
        findings, records = [], []
        for track in observation.tracks.values():
            path = self.measure(track, observation)
            verdict, evaluable, groups = self._judge(path, durations)
            records.append({"track_id": path.track_id, "verdict": verdict, "evaluable_windows": evaluable})
            findings += [self._finding(path, group) for group in groups]
        findings.sort(key=lambda f: f.trigger_frame)
        return RuleResult(findings, records, self.settings.model_dump())

    def measure(self, track: Track, observation: Observation) -> TrackPath:
        boxes = np.array([p.box.as_list() for p in track.points], dtype=np.float64)
        frames = np.array([p.frame_index for p in track.points])
        centers = (boxes[:, 0] + boxes[:, 2]) / 2
        motion, m = observation.motion, self.settings.border_margin
        return TrackPath(
            track_id=str(track.track_id),
            class_name=track.class_name,
            frames=frames,
            bottom=np.vstack([motion.to_reference(f, [[x, y]]) for f, x, y in zip(frames, centers, boxes[:, 3])]),
            top=np.vstack([motion.to_reference(f, [[x, y]]) for f, x, y in zip(frames, centers, boxes[:, 1])]),
            heights=boxes[:, 3] - boxes[:, 1],
            clipped=(boxes[:, 0] <= m) | (boxes[:, 1] <= m) | (boxes[:, 2] >= observation.width - m) | (boxes[:, 3] >= observation.height - m),
        )

    def windows(self, path: TrackPath, durations: Durations) -> tuple[list[dict], bool]:
        """Every window whose midpoint is in a region, with its measurements, and whether the track was ever seen in one. No thresholds apply here."""
        frames = path.frames
        kernel = (durations.smoothing, 1)
        bottom = median_filter(path.bottom, size=kernel, mode="nearest")
        top = median_filter(path.top, size=kernel, mode="nearest")
        regions = self.scene.regions
        seen_in_region = any(point_in_polygon(r.polygon, p) for r in regions for p in bottom[::5])
        windows = []
        for i, start_frame in enumerate(frames):
            j = int(np.searchsorted(frames, start_frame + durations.window))
            if j >= len(frames) or frames[j] != start_frame + durations.window:
                continue
            midpoint = (bottom[i] + bottom[j]) / 2
            region = next((r for r in regions if point_in_polygon(r.polygon, midpoint)), None)
            if region is None:
                continue
            allowed = allowed_direction(region, midpoint, self.scene.vanishing_point)
            displacement = bottom[j] - bottom[i]
            distance = float(np.linalg.norm(displacement))
            along = float(-(displacement @ allowed))
            box_height = float(np.median(path.heights[i:j + 1]))
            windows.append({
                "start_frame": int(frames[i]),
                "end_frame": int(frames[j]),
                "region_id": region.id,
                "opposing_px": along,
                "opposing_heights": along / box_height,
                "top_opposing_px": float(-((top[j] - top[i]) @ allowed)),
                "angle_to_opposite_deg": float(np.degrees(np.arccos(np.clip(along / distance, -1, 1)))) if distance > 0 else 180.0,
                "tolerance_deg": region.tolerance_deg,
                "observed_fraction": (j - i + 1) / (durations.window + 1),
                "clipped": bool(path.clipped[i:j + 1].any()),
            })
        return windows, seen_in_region

    def usable(self, window: dict) -> bool:
        """C5: the window can be trusted."""
        return window["observed_fraction"] >= self.settings.min_observed_fraction and not window["clipped"]

    def failed_conditions(self, window: dict) -> list[str]:
        """Which conditions a window fails; an empty list means it passes."""
        s = self.settings
        failed = []
        if window["angle_to_opposite_deg"] > window["tolerance_deg"]:
            failed.append("C2 direction")
        if window["opposing_heights"] < s.min_opposing_heights:
            failed.append("C3 distance")
        if window["opposing_px"] <= 0 or window["top_opposing_px"] < s.min_top_edge_share * window["opposing_px"]:
            failed.append("C4 top edge")
        if not self.usable(window):
            failed.append("C5 quality")
        return failed

    def _judge(self, path: TrackPath, durations: Durations) -> tuple[str, int, list[list[dict]]]:
        """The track's verdict, its number of usable windows, and its passing windows grouped into events.

        ``not_evaluable`` is kept apart from ``no_violation``: "we could not check" differs from "we found nothing".
        """
        windows, seen_in_region = self.windows(path, durations)
        evaluable = [w for w in windows if self.usable(w)]
        passing = [w for w in evaluable if not self.failed_conditions(w)]
        if passing:
            verdict = CANDIDATE
        elif evaluable:
            verdict = NO_VIOLATION
        elif seen_in_region or windows:
            verdict = NOT_EVALUABLE
        else:
            verdict = NOT_IN_REGION
        return verdict, len(evaluable), self._merge(passing, durations)

    def _merge(self, passing: list[dict], durations: Durations) -> list[list[dict]]:
        """Overlapping passing windows of one region form one event."""
        groups: list[list[dict]] = []
        for window in sorted(passing, key=lambda w: w["start_frame"]):
            if groups and window["region_id"] == groups[-1][-1]["region_id"] and window["start_frame"] <= groups[-1][-1]["end_frame"] + durations.merge_gap:
                groups[-1].append(window)
            else:
                groups.append([window])
        return groups

    def _finding(self, path: TrackPath, group: list[dict]) -> Finding:
        first, last = group[0], group[-1]
        best = max(group, key=lambda w: w["opposing_heights"])
        start, trigger, end = first["start_frame"], first["end_frame"], last["end_frame"]
        return Finding(
            rule=self.name,
            track_id=path.track_id,
            class_name=path.class_name,
            start_frame=start,
            trigger_frame=trigger,
            end_frame=end,
            why=(f"Moved against the allowed direction of {first['region_id']} by {best['opposing_heights']:.2f} box heights "
                 f"in about {self.settings.window_seconds:.1f} s, {best['angle_to_opposite_deg']:.0f} degrees from the opposite direction."),
            evidence={
                "region_id": first["region_id"],
                "opposing_heights": round(best["opposing_heights"], 3),
                "opposing_px": round(best["opposing_px"], 1),
                "top_opposing_px": round(best["top_opposing_px"], 1),
                "angle_to_opposite_deg": round(best["angle_to_opposite_deg"], 1),
                "evidence_frames": [start, trigger, end],
            },
        )
```

`src/tvd/rules/__init__.py`:

```python
"""Violation rules. Importing this package registers every rule in it."""

from tvd.rules import wrong_way  # noqa: F401  Registers "wrong_way".
from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult

__all__ = ["RULES", "BaseRule", "Finding", "Observation", "RuleResult"]
```

`configs/tvd/rules.yaml`:

```yaml
# The settings of each rule, for every camera. A camera file can change any of them for one camera,
# under "rules: <rule>: settings:". Changing a value means the scores of Notebooks 05, 05b and 09 no
# longer apply: every result records the settings that produced it. The red-light rule joins in Task 11.

wrong_way:
  version: 1
  window_seconds: 2.0           # how long a vehicle must move against the direction; chosen in Notebook 05
  smoothing_seconds: 0.15
  merge_gap_seconds: 0.25       # passing windows closer than this are one event
  min_opposing_heights: 1.0     # C3: it must move at least this many box heights against the direction
  min_top_edge_share: 0.25      # C4: the top edge must move at least this share of that distance
  min_observed_fraction: 0.8    # C5: share of the window's frames in which the vehicle is seen
  border_margin: 3              # px
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `101 passed`.

`test_the_same_results_as_the_notebooks[True]` plays the 136 tracks of `video-test.mp4` backwards and must find the same wrong-way candidates as the notebook's rule. If it fails, print the first track whose verdict differs, and compare its windows with `notebook.track_windows`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_wrong_way.py src/tvd/rules/wrong_way.py src/tvd/rules/__init__.py configs/tvd/rules.yaml
git commit -m "Add the wrong-way rule"
```

### Task 10: Joining track fragments

**Why:** Version 2 of the red-light rule joins two tracks that are plainly one vehicle: one ends where the other begins, at the same size. It is tracking logic, so it lives in `tracking/`, as a pure function. The limits are strict on purpose: joining two vehicles could give one of them the other's plate.

**Files:**
- Create: `tests/tvd/test_tvd_fragments.py`
- Create: `src/tvd/tracking/fragments.py`

**Interfaces:**
- Consumes: `TrackPoint` (Task 1).
- Produces: `join_fragments(tracks: dict[str, list[TrackPoint]], max_gap_frames, max_jump_heights, max_size_ratio, box_of=None) -> dict[str, list[TrackPoint]]`; a joined id is `"<first>+<second>"`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_fragments.py`:

```python
import pytest
from notebook_data import OUTPUTS, history_from_rows, load_json

from tvd.domain import Box
from tvd.tracking.fragments import join_fragments
from tvd.tracking.history import TrackPoint

STORED_TRACKS = OUTPUTS / "05b_red_light_violation" / "tracks.json"
LIMITS = {"max_gap_frames": 6, "max_jump_heights": 0.1, "max_size_ratio": 1.25}


def points(first: int, last: int, x: float = 100, height: float = 40) -> list[TrackPoint]:
    """A vehicle standing still, seen from frame ``first`` to ``last``."""
    return [TrackPoint(k, Box(x, 100, x + 50, 100 + height), 0.9, "car") for k in range(first, last + 1)]


def test_a_track_that_continues_another_is_joined():
    joined = join_fragments({"3": points(0, 10), "8": points(12, 20)}, **LIMITS)
    assert list(joined) == ["3+8"] and [p.frame_index for p in joined["3+8"]] == list(range(11)) + list(range(12, 21))


@pytest.mark.parametrize("second, why", [
    (points(20, 30), "starts too long after"),
    (points(12, 20, x=110), "starts too far away"),
    (points(12, 20, height=60), "is a different size"),
])
def test_tracks_that_may_be_two_vehicles_are_kept_apart(second, why):
    assert list(join_fragments({"3": points(0, 10), "8": second}, **LIMITS)) == ["3", "8"], why


def test_the_closest_continuation_is_joined_first():
    joined = join_fragments({"1": points(0, 10), "2": points(11, 20, x=103), "3": points(11, 20, x=101)}, **LIMITS)
    assert sorted(joined) == ["1+3", "2"]


@pytest.mark.skipif(not STORED_TRACKS.is_file(), reason="Needs the outputs of Notebook 05b, as on the GPU server.")
def test_the_same_joins_as_the_notebooks():
    from traffic_violation_detection.tracking import join_track_fragments

    for clip, tracks in load_json(STORED_TRACKS).items():
        expected = join_track_fragments(tracks, **LIMITS)
        ours = join_fragments({str(t.track_id): t.points for t in history_from_rows(tracks).tracks.values()}, **LIMITS)
        assert {k: [p.frame_index for p in v] for k, v in ours.items()} == {k: [int(r[0]) for r in v] for k, v in expected.items()}, clip
        assert list(ours) == list(expected), clip
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_fragments.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.tracking.fragments'`

- [ ] **Step 3: Write the code**

`src/tvd/tracking/fragments.py`:

```python
"""Joining two tracks that are plainly one vehicle under two ids."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tvd.tracking.history import TrackPoint

BoxOf = Callable[[TrackPoint], list[float]]


def join_fragments(
    tracks: dict[str, list[TrackPoint]],
    max_gap_frames: int,
    max_jump_heights: float,
    max_size_ratio: float,
    box_of: BoxOf | None = None,
) -> dict[str, list[TrackPoint]]:
    """Join tracks where one ends and another begins at the same place and size. A joined id is ``"<first>+<second>"``.

    The limits are strict on purpose: joining two vehicles could give one the other's plate, while a
    vehicle left in two parts only costs a missed case. ``box_of`` gives a point's box, in reference
    pixels when the camera moves. The closest continuation is joined first, until none is left.
    """
    box_of = box_of or (lambda point: point.box.as_list())

    def bottom_and_height(point: TrackPoint) -> tuple[float, float, float]:
        x1, y1, x2, y2 = box_of(point)
        return (x1 + x2) / 2, y2, y2 - y1

    def continuation(first: list[TrackPoint], second: list[TrackPoint]) -> float | None:
        """How far the second track starts from the end of the first, in box heights, or None if it does not continue it."""
        starts_later = second[0].frame_index > first[0].frame_index and second[-1].frame_index > first[-1].frame_index
        if not (starts_later and abs(second[0].frame_index - first[-1].frame_index) <= max_gap_frames):
            return None
        x1, y1, height1 = bottom_and_height(first[-1])
        x2, y2, height2 = bottom_and_height(second[0])
        jump = float(np.hypot(x2 - x1, y2 - y1)) / max(height1, 1)
        return jump if jump <= max_jump_heights and max(height1 / height2, height2 / height1) <= max_size_ratio else None

    joined = {track_id: list(points) for track_id, points in tracks.items()}
    while True:
        pairs = [(continuation(joined[a], joined[b]), a, b) for a in joined for b in joined if a != b]
        pairs = [p for p in pairs if p[0] is not None]
        if not pairs:
            return joined
        _, first, second = min(pairs)
        later = joined.pop(second)
        earlier = joined.pop(first)
        joined[f"{first}+{second}"] = [p for p in earlier if p.frame_index < later[0].frame_index] + later
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `107 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_fragments.py src/tvd/tracking/fragments.py
git commit -m "Join the fragments of one vehicle's track"
```

### Task 11: Red-light rule

**Why:** The red-light rule of Notebook 05b, rewritten as a `BaseRule`. It builds its signal readers from its scene, through `SIGNAL_READERS`, and hands its `SignalMonitor` to the pass. `judge(observation, signals)` takes the signal states as an argument, so the notebook test can give it the states Notebook 05b saved. That test runs all 18 clips of `rlr_occasions.mp4`, with rule versions 1 and 2, with a fixed camera and with a made-up camera shake: every crossing record must be **equal** to the notebook's.

**Files:**
- Create: `tests/tvd/test_tvd_red_light.py`
- Create: `src/tvd/rules/red_light.py`
- Replace: `src/tvd/rules/__init__.py`
- Replace: `configs/tvd/rules.yaml`

**Interfaces:**
- Consumes: `BaseRule` and its types (Task 8); `SignalMonitor`, `SignalTimeline`, `SIGNAL_READERS` (Tasks 6 and 7); `join_fragments` (Task 10); `majority_class` (Task 1); `polyline_y` (Task 3); `CameraMotion` (Task 4).
- Produces: `RedLightRule(scene: dict, settings: dict)` as `"red_light"`, with `frame_observers()`, `evaluate(observation)`, `judge(observation, signals)`, `crossing(...)`, `deduplicate(records, fps)`, `lane_of(x)`; `SignalHeads`, `RedLightScene`, `RedLightSettings`; `movement_after`, `hidden_entry`, `reference_box`; verdicts `candidate`, `signal_not_red`, `stopped_at_line`, `right_turn_on_red`, `not_evaluable_signal`, `not_evaluable_border`, `not_evaluable_occluded_line`, `not_evaluable_ended`. `configs/tvd/rules.yaml` complete.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_red_light.py`:

```python
import numpy as np
import pytest
import yaml
from conftest import ROOT
from notebook_data import OUTPUTS, RED_LIGHT_SCENE, history_from_rows, load_json, red_light_section

from tvd.config import ConfigError
from tvd.domain import Box, Frame, TrackedObject
from tvd.motion import CameraMotion
from tvd.rules import RULES, Observation
from tvd.rules.red_light import RedLightRule
from tvd.signals import SignalTimeline
from tvd.tracking.history import TrackHistory

SETTINGS = yaml.safe_load((ROOT / "configs" / "tvd" / "rules.yaml").read_text())["red_light"]
VERSION_1 = {**SETTINGS, "version": 1, "join_fragments": False, "estimate_hidden_entry": False}
# A horizontal stop line at y = 200 on a 640x360 image. Traffic goes up the image, away from the camera.
SCENE = {
    "stop_line": [[0, 200], [640, 200]], "lane_dividers": [100, 540], "lane_direction_deg": -90, "traffic_direction": "away_from_camera",
    "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}], "left": []},
}
STORED = OUTPUTS / "05b_red_light_violation"
needs_stored_clips = pytest.mark.skipif(not (STORED / "tracks.json").is_file(), reason="Needs the outputs of Notebook 05b, as on the GPU server.")


def car_up(frames: int = 60, x: float = 300, stop_at: float | None = None, turn_right_after: int | None = None) -> list[tuple[float, float]]:
    """Bottom-left corners of a 40x30 px car going up the image by 4 px per frame."""
    path = []
    for k in range(frames):
        y2 = 330 - 4 * k if stop_at is None else max(330 - 4 * k, stop_at)
        dx = 0
        if turn_right_after is not None and k > turn_right_after:
            y2, dx = 330 - 4 * turn_right_after, 6 * (k - turn_right_after)
        path.append((x + dx, y2))
    return path


def judge(path, state: str = "red", settings: dict = SETTINGS, scene: dict = SCENE):
    history = TrackHistory()
    for k, (x1, y2) in enumerate(path):
        history.add(k, [TrackedObject(7, Box(x1, y2 - 30, x1 + 40, y2), 2, "car", 0.9)])
    signals = SignalTimeline({"through": {k: state for k in range(len(path))}, "left": {k: "unknown" for k in range(len(path))}})
    return RedLightRule(scene, settings).judge(Observation(history.tracks, CameraMotion(), 24.0, 640, 360, 0, len(path)), signals)


def test_the_rule_is_registered():
    assert "red_light" in RULES.names()


def test_entering_on_red_and_passing_the_line_is_a_candidate():
    result = judge(car_up())
    assert result.records == [{
        "track_id": "7", "class_name": "car", "frame": 31, "x": 320.0, "lane": "through", "signal_group": "through", "signal_state": "red",
        "state_for_frames": 31, "movement": "straight", "passed_line": True, "entry_seen": True, "entry_estimated": False, "verdict": "candidate",
        "box_xyxy": [300.0, 176.0, 340.0, 206.0],
    }]
    (finding,) = result.findings
    assert (finding.start_frame, finding.trigger_frame, finding.end_frame) == (0, 31, 59)
    assert finding.why == "Entered in lane 'through' while the through signal was red (1.3 s after it was first seen in that state), and continued past the stop line."


@pytest.mark.parametrize("state, verdict", [("green", "signal_not_red"), ("yellow", "signal_not_red"), ("unknown", "not_evaluable_signal"), ("conflict", "not_evaluable_signal")])
def test_the_signal_decides(state, verdict):
    assert judge(car_up(), state).verdict_counts() == {verdict: 1}


def test_a_vehicle_that_stops_on_the_line_is_not_a_candidate():
    assert judge(car_up(80, stop_at=206)).verdict_counts() == {"stopped_at_line": 1}


def test_a_right_turn_on_red_is_not_a_candidate():
    assert judge(car_up(60, turn_right_after=35)).verdict_counts() == {"right_turn_on_red": 1}


def test_a_box_at_the_image_border_cannot_be_judged():
    assert judge(car_up(x=598)).verdict_counts() == {"not_evaluable_border": 1}


def test_the_left_turn_lane_follows_the_left_signal():
    assert judge(car_up(x=30)).records[0]["signal_group"] == "left"  # The left group has no heads here: its state is unknown.


def test_a_vehicle_that_never_crosses_has_no_record():
    assert judge(car_up(20)).records == []


def test_the_rule_reads_the_signals_itself_during_the_pass():
    rule = RedLightRule(SCENE, SETTINGS)
    (monitor,) = rule.frame_observers()
    history = TrackHistory()
    for k, (x1, y2) in enumerate(car_up()):
        monitor.observe(Frame(k, k / 24, np.zeros((360, 640, 3), np.uint8)), [])
        history.add(k, [TrackedObject(7, Box(x1, y2 - 30, x1 + 40, y2), 2, "car", 0.9)])
    assert rule.evaluate(Observation(history.tracks, CameraMotion(), 24.0, 640, 360, 0, 60)).verdict_counts() == {"candidate": 1}


@pytest.mark.parametrize("change, message", [
    ({"settings": {"pass_margin": -1}}, "settings.pass_margin: Input should be greater than or equal to 0"),
    ({"scene": {"stop_line": [[640, 200], [0, 200]]}}, "scene.stop_line: Value error, x must increase from left to right"),
    ({"scene": {"signals": {"reader": "laser", "through": [{}]}}}, "Unknown signal reader 'laser'"),
])
def test_a_wrong_setting_is_named(change, message):
    with pytest.raises(ValueError, match=message):  # ConfigError is a ValueError.
        RedLightRule({**SCENE, **change.get("scene", {})}, {**SETTINGS, **change.get("settings", {})})


def notebook_inputs(clip_name: str):
    scene, stored_tracks, stored_signals = load_json(RED_LIGHT_SCENE), load_json(STORED / "tracks.json"), load_json(STORED / "signals.json")
    states = {group: {int(k): v for k, v in by_frame.items()} for group, by_frame in stored_signals[clip_name].items()}
    return scene, scene["clips"][clip_name], stored_tracks[clip_name], states


@needs_stored_clips
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("moving", [False, True], ids=["fixed camera", "moving camera"])
def test_the_same_results_as_the_notebooks_on_every_clip(version, moving):
    from traffic_violation_detection import red_light as notebook

    rule_v = notebook.RULE if version == 2 else notebook.RULE_V1
    settings = SETTINGS if version == 2 else VERSION_1
    for clip_name in load_json(RED_LIGHT_SCENE)["clips"]:
        scene, clip, tracks, states = notebook_inputs(clip_name)
        # A made-up camera shake, so that the moving-camera code is compared too.
        params = np.array([[1.0, 0.0005 * np.sin(k / 40), 3 * np.sin(k / 25), 2 * np.cos(k / 35)] for k in range(clip["end"] + 1)]) if moving else None
        joined = notebook.join_clip_tracks(tracks, params, rule_v)
        expected = notebook.evaluate_clip(joined, clip, states, scene["lane_direction_deg"], 1280, 720, 30.0, rule_v, params)

        section = red_light_section(scene, clip_name)
        rule = RedLightRule(section["scene"], {**settings, **section["settings"]})
        motion = CameraMotion() if params is None else CameraMotion(params)
        observation = Observation(history_from_rows(tracks).tracks, motion, 30.0, 1280, 720, clip["start"], clip["end"])
        result = rule.judge(observation, SignalTimeline(states))

        assert result.records == expected, clip_name
        events = [notebook.build_event(r, clip_name, "rlr", 30.0, clip["start"], "camera", rule_v) for r in expected if r["verdict"] == "candidate"]
        assert [(f.track_id, f.class_name, f.trigger_frame, f.why, f.evidence) for f in result.findings] == [
            (e["track_id"], e["class_name"], e["trigger_frame"], e["why"], {**e["evidence"], "signal_source": "camera"}) for e in events
        ], clip_name
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_red_light.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.rules.red_light'`

- [ ] **Step 3: Write the code**

Replace `src/tvd/rules/__init__.py` so that it registers both rules, and replace `configs/tvd/rules.yaml` with its complete version: the red-light section is added above the wrong-way one, which does not change.

`src/tvd/rules/red_light.py`:

```python
"""Red-light rule, developed in Notebook 05b. Its output is a candidate for review, never a verdict.

A crossing of the stop line is a candidate only if all of these hold:

- the signal of the vehicle's lane is readable and red when the vehicle enters;
- the vehicle travels on past the line (one that stops on the line is not flagged);
- it does not turn right (a right turn on red may be legal);
- its box does not touch the image border, and it crosses the part of the line that can be judged.

When information is missing, the crossing is not evaluable, never a violation. Version 2 of the rule
(``join_fragments`` and ``estimate_hidden_entry``) also handles a vehicle whose track breaks.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator
from scipy.ndimage import median_filter

from tvd.config import StrictModel, validate
from tvd.geometry import polyline_y
from tvd.motion import CameraMotion
from tvd.observers import FrameObserver
from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult
from tvd.signals import SIGNAL_READERS, UNKNOWN, SignalMonitor, SignalTimeline
from tvd.tracking.fragments import join_fragments
from tvd.tracking.history import TrackPoint, majority_class

SIGNAL_GROUPS = ("through", "left")

CANDIDATE = "candidate"
COMPLIANT = "signal_not_red"
STOPPED = "stopped_at_line"
RIGHT_TURN = "right_turn_on_red"
UNKNOWN_SIGNAL = "not_evaluable_signal"
AT_BORDER = "not_evaluable_border"
OUT_OF_RANGE = "not_evaluable_occluded_line"
ENDED = "not_evaluable_ended"


class SignalHeads(StrictModel):
    """The signal heads of each group, read by one reader. ``options`` go to every head's reader, for example ``half``."""

    reader: str
    options: dict[str, Any] = Field(default_factory=dict)
    through: list[dict[str, Any]] = Field(min_length=1)
    left: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("options", mode="before")
    @classmethod
    def no_options_when_left_empty(cls, value: Any) -> Any:
        return {} if value is None else value


class RedLightScene(StrictModel):
    """Where the stop line, the lanes and the signals are. With a moving camera, the line is drawn on the reference frame."""

    stop_line: list[tuple[float, float]] = Field(min_length=2)
    lane_dividers: list[float] = Field(min_length=1)  # Left of the first: the left-turn lane. Right of the last: the right-turn lane.
    lane_direction_deg: float                         # Direction of travel in the image, in degrees, with y pointing down.
    traffic_direction: Literal["away_from_camera", "toward_camera"]
    evaluable_x_max: float | None = None              # Crossings right of this cannot be judged, for example behind a pole.
    signal_source: str = "camera"
    signals: SignalHeads

    @field_validator("stop_line", "lane_dividers")
    @classmethod
    def increasing_x(cls, value: list) -> list:
        xs = [v[0] if isinstance(v, tuple) else v for v in value]
        if xs != sorted(xs):
            raise ValueError("x must increase from left to right")
        return value


class RedLightSettings(StrictModel):
    version: int
    entry_offset: float = Field(ge=0)          # The front wheels: this fraction of the box height above its bottom.
    pass_margin: float = Field(ge=0)           # The rear must pass the line by this many box heights.
    min_track_frames: int = Field(ge=2)
    smoothing_frames: int = Field(ge=1)
    min_frames_to_decide: int = Field(ge=1)    # Fewer frames left than this: stopped and still moving cannot be told apart.
    still_box_heights: float = Field(ge=0)     # Stopped: moved less than this many box heights over those frames.
    min_heading_px: float = Field(ge=0)        # A shorter movement after the line: heading unknown.
    turn_angle_deg: float = Field(gt=0, lt=180)
    border_margin: float = Field(ge=0)
    duplicate_seconds: float = Field(ge=0)
    duplicate_px: float = Field(ge=0)
    join_fragments: bool
    join_max_gap_frames: int = Field(ge=0)
    join_max_jump_heights: float = Field(ge=0)
    join_max_size_ratio: float = Field(ge=1)
    estimate_hidden_entry: bool
    hidden_fit_frames: int = Field(ge=5)
    hidden_max_frames: int = Field(ge=0)
    hidden_margin_frames: int = Field(ge=0)
    hidden_min_speed: float = Field(ge=0)
    signal_smoothing_frames: int = Field(ge=1)


def movement_after(displacement, lane_direction_deg: float, min_heading_px: float, turn_angle_deg: float) -> str:
    """Movement relative to the lane direction: image axes mislead, because lanes run diagonally in the image."""
    dx, dy = displacement
    if np.hypot(dx, dy) < min_heading_px:
        return "unknown"
    heading = np.degrees(np.arctan2(dy, dx))
    turn = (heading - lane_direction_deg + 180) % 360 - 180
    if turn > turn_angle_deg:
        return "right"
    return "left" if turn < -turn_angle_deg else "straight"


def hidden_entry(frames, bottom, front_margin, box_heights, lane_direction_deg: float, s: RedLightSettings) -> tuple[int, float] | None:
    """The frame and column where a vehicle crossed the line, for a track that starts past it.

    The track's first frames are followed back to the line. None means no estimate: the vehicle is
    waiting rather than leaving the line, it is not moving along the lane, or the line is too far back.
    """
    count = min(s.hidden_fit_frames, len(frames))
    if count < 5:
        return None
    speed = float(np.polyfit(frames[:count], front_margin[:count], 1)[0])  # Negative: moving further past the line.
    if speed >= -s.hidden_min_speed * box_heights[0]:
        return None
    frames_back = float(front_margin[0]) / speed
    if frames_back > s.hidden_max_frames:
        return None
    if movement_after(bottom[count - 1] - bottom[0], lane_direction_deg, 0, s.turn_angle_deg) != "straight":
        return None
    sideways = float(np.polyfit(frames[:count], bottom[:count, 0], 1)[0])
    return int(round(frames[0] - frames_back)), float(bottom[0, 0] - sideways * frames_back)


def reference_box(point: TrackPoint, motion: CameraMotion) -> list[float]:
    """A box moved into reference pixels. It keeps its size, and its bottom centre decides where it lands."""
    x1, y1, x2, y2 = point.box.as_list()
    x, y = motion.to_reference(point.frame_index, [[(x1 + x2) / 2, y2]])[0]
    width, height = x2 - x1, y2 - y1
    return [float(x - width / 2), float(y - height), float(x + width / 2), float(y)]


@RULES.register("red_light")
class RedLightRule(BaseRule):
    name = "red_light"

    def __init__(self, scene: dict[str, Any], settings: dict[str, Any]) -> None:
        self.scene = validate(RedLightScene, scene, "scene")
        self.settings = validate(RedLightSettings, settings, "settings")
        heads = self.scene.signals
        readers = {group: [SIGNAL_READERS.create(heads.reader, **heads.options, **head) for head in getattr(heads, group)] for group in SIGNAL_GROUPS}
        self._monitor = SignalMonitor(readers, self.settings.signal_smoothing_frames)

    def frame_observers(self) -> list[FrameObserver]:
        return [self._monitor]

    def evaluate(self, observation: Observation) -> RuleResult:
        return self.judge(observation, self._monitor.timeline())

    def judge(self, observation: Observation, signals: SignalTimeline) -> RuleResult:
        """Judge every track against the stop line, with signal states read elsewhere."""
        s, motion = self.settings, observation.motion
        tracks = {str(t.track_id): t.points for t in observation.tracks.values()}
        if s.join_fragments:
            box_of = None if motion.is_fixed else (lambda point: reference_box(point, motion))
            tracks = join_fragments(tracks, s.join_max_gap_frames, s.join_max_jump_heights, s.join_max_size_ratio, box_of)
        records = [self.crossing(track_id, points, observation, signals) for track_id, points in tracks.items()]
        records = self.deduplicate([r for r in records if r], observation.fps)
        findings = [self._finding(r, tracks[r["track_id"]], observation.fps) for r in records if r["verdict"] == CANDIDATE]
        return RuleResult(findings, records, s.model_dump())

    def lane_of(self, x: float) -> str:
        dividers = self.scene.lane_dividers
        if x < dividers[0]:
            return "left_turn"
        return "right_turn" if x >= dividers[-1] else "through"

    def crossing(self, track_id: str, points: list[TrackPoint], observation: Observation, signals: SignalTimeline) -> dict | None:
        """How one track crossed the stop line, with the rule's verdict, or None if it never crosses.

        Coming toward the camera, a box's bottom edge is already the vehicle's front, so such a camera
        sets ``entry_offset`` to 0. An estimated entry is judged only if one signal state lasted from
        before it until the vehicle was seen.
        """
        s, scene = self.settings, self.scene
        if len(points) < s.min_track_frames:
            return None
        frames = np.array([p.frame_index for p in points])
        frame_boxes = np.array([p.box.as_list() for p in points], dtype=np.float64)
        boxes = frame_boxes if observation.motion.is_fixed else np.array([reference_box(p, observation.motion) for p in points], dtype=np.float64)
        centers = (boxes[:, 0] + boxes[:, 2]) / 2
        kernel = s.smoothing_frames
        bottom = median_filter(np.c_[centers, boxes[:, 3]], size=(kernel, 1), mode="nearest")
        box_heights = median_filter(boxes[:, 3] - boxes[:, 1], size=kernel, mode="nearest")
        line = polyline_y(scene.stop_line, bottom[:, 0])
        side = -1.0 if scene.traffic_direction == "toward_camera" else 1.0  # Which side of the line is "before" it.
        rear_margin = side * (bottom[:, 1] - line)                         # Positive: the rear is still before the line.
        front_margin = rear_margin - s.entry_offset * box_heights          # Positive: the front is still before the line.

        front_cross = np.flatnonzero((front_margin[:-1] > 0) & (front_margin[1:] <= 0))
        rear_cross = np.flatnonzero((rear_margin[:-1] > 0) & (rear_margin[1:] <= 0))
        estimated = None
        if len(front_cross):
            index, entry_seen = int(front_cross[0]), True
        elif front_margin[0] <= 0 and len(rear_cross):
            # The track starts with the front already past the line: a fragment of an earlier track of this vehicle.
            index, entry_seen = int(rear_cross[0]), False
        elif s.estimate_hidden_entry and front_margin[0] <= 0:
            estimated = hidden_entry(frames, bottom, front_margin, box_heights, scene.lane_direction_deg, s)
            if estimated is None or estimated[0] - s.hidden_margin_frames < observation.first_frame:
                return None
            index, entry_seen = -1, False
        else:
            return None

        at = index + 1
        frame, x = estimated if estimated else (int(frames[at]), float(bottom[at, 0]))
        stop_xs = [p[0] for p in scene.stop_line]
        if not (stop_xs[0] - 30 <= x <= stop_xs[-1] + 30):
            return None

        lane = self.lane_of(x)
        group = "left" if lane == "left_turn" else "through"
        state = signals.state(group, frame)
        if estimated:
            seen = {signals.state(group, k) for k in range(frame - s.hidden_margin_frames, int(frames[0]) + 1)}
            state = seen.pop() if len(seen) == 1 else "conflict"
        state_since = frame
        while signals.get(group, state_since - 1) == state:
            state_since -= 1
        movement = movement_after(bottom[-1] - bottom[at], scene.lane_direction_deg, s.min_heading_px, s.turn_angle_deg)
        box = frame_boxes[at]
        margin = s.border_margin
        at_border = bool(box[0] <= margin or box[2] >= observation.width - margin or box[3] >= observation.height - margin)
        later = rear_margin[at:]
        passed = bool(later.min() <= -s.pass_margin * box_heights[at])
        decide = s.min_frames_to_decide
        recent_motion = float(np.hypot(*(bottom[-1] - bottom[-min(decide, len(later))])))
        still = recent_motion <= s.still_box_heights * box_heights[at]
        evaluable_x_max = observation.width if scene.evaluable_x_max is None else scene.evaluable_x_max

        if at_border:
            verdict = AT_BORDER
        elif x > evaluable_x_max:
            verdict = OUT_OF_RANGE
        elif state in (UNKNOWN, "conflict"):
            verdict = UNKNOWN_SIGNAL
        elif state != "red":
            verdict = COMPLIANT
        elif movement == "right":
            verdict = RIGHT_TURN
        elif passed:
            verdict = CANDIDATE
        elif len(later) < decide or observation.end_frame - frame < decide or not still:
            verdict = ENDED  # A vehicle still moving when its track ends may yet pass the line: that is not a stop.
        else:
            verdict = STOPPED
        return {
            "track_id": track_id, "class_name": majority_class(points), "frame": frame, "x": round(x, 1),
            "lane": lane, "signal_group": group, "signal_state": state, "state_for_frames": frame - state_since, "movement": movement,
            "passed_line": passed, "entry_seen": entry_seen, "entry_estimated": estimated is not None, "verdict": verdict,
            "box_xyxy": [round(float(v), 1) for v in box],
        }

    def deduplicate(self, records: list[dict], fps: float) -> list[dict]:
        """Merge candidates close in time and space: tracker fragments of one vehicle."""
        s = self.settings
        kept = []
        for record in sorted(records, key=lambda r: r["frame"]):
            if record["verdict"] == CANDIDATE and any(
                k["verdict"] == CANDIDATE and (record["frame"] - k["frame"]) / fps <= s.duplicate_seconds and abs(record["x"] - k["x"]) <= s.duplicate_px
                for k in kept
            ):
                continue
            kept.append(record)
        return kept

    def _finding(self, record: dict, points: list[TrackPoint], fps: float) -> Finding:
        entered = "Estimated to have entered, while hidden from view," if record["entry_estimated"] else "Entered"
        return Finding(
            rule=self.name,
            track_id=record["track_id"],
            class_name=record["class_name"],
            start_frame=points[0].frame_index,
            trigger_frame=record["frame"],
            end_frame=points[-1].frame_index,
            why=(f"{entered} in lane '{record['lane']}' while the {record['signal_group']} signal was red "
                 f"({record['state_for_frames'] / fps:.1f} s after it was first seen in that state), and continued past the stop line."),
            evidence={
                **{k: record[k] for k in ("x", "lane", "signal_group", "signal_state", "movement", "entry_seen", "entry_estimated", "box_xyxy")},
                "signal_source": self.scene.signal_source,
            },
        )
```

`src/tvd/rules/__init__.py`:

```python
"""Violation rules. Importing this package registers every rule in it."""

from tvd.rules import red_light, wrong_way  # noqa: F401  Registers "red_light" and "wrong_way".
from tvd.rules.base import RULES, BaseRule, Finding, Observation, RuleResult

__all__ = ["RULES", "BaseRule", "Finding", "Observation", "RuleResult"]
```

`configs/tvd/rules.yaml`:

```yaml
# The settings of each rule, for every camera. A camera file can change any of them for one camera,
# under "rules: <rule>: settings:". Changing a value means the scores of Notebooks 05, 05b and 09 no
# longer apply: every result records the settings that produced it.

red_light:
  version: 2                    # 1 is the rule scored in Notebook 05b; 2 adds the two settings marked (v2)
  entry_offset: 0.30            # the front wheels: this fraction of the box height above its bottom
  pass_margin: 0.5              # the rear must pass the line by this many box heights
  min_track_frames: 8
  smoothing_frames: 5           # median filter over a track's positions
  min_frames_to_decide: 15      # fewer frames left than this: stopped and still moving cannot be told apart
  still_box_heights: 0.15       # stopped: moved less than this many box heights over those frames
  min_heading_px: 60            # a shorter movement after the line: heading unknown
  turn_angle_deg: 35            # more than this from the lane direction is a turn
  border_margin: 3              # px: a box this close to the image border may be cut off
  duplicate_seconds: 1.0        # two candidates this close in time ...
  duplicate_px: 120             # ... and space are one vehicle
  join_fragments: true          # (v2) join two tracks that are plainly one vehicle
  join_max_gap_frames: 6
  join_max_jump_heights: 0.1
  join_max_size_ratio: 1.25
  estimate_hidden_entry: true   # (v2) estimate when a vehicle hidden at the line crossed it
  hidden_fit_frames: 10         # the track's first frames, followed back to the line
  hidden_max_frames: 30         # an entry further back than this is too uncertain
  hidden_margin_frames: 13      # the signal must hold this long before the estimated entry: about 0.5 s
  hidden_min_speed: 0.01        # box heights per frame; slower is a vehicle waiting, not leaving the line
  signal_smoothing_frames: 7    # majority vote over this many frames, against flicker

wrong_way:
  version: 1
  window_seconds: 2.0           # how long a vehicle must move against the direction; chosen in Notebook 05
  smoothing_seconds: 0.15
  merge_gap_seconds: 0.25       # passing windows closer than this are one event
  min_opposing_heights: 1.0     # C3: it must move at least this many box heights against the direction
  min_top_edge_share: 0.25      # C4: the top edge must move at least this share of that distance
  min_observed_fraction: 0.8    # C5: share of the window's frames in which the vehicle is seen
  border_margin: 3              # px
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `126 passed`.

`test_the_same_results_as_the_notebooks_on_every_clip` runs 4 variants of 18 clips in a few seconds. It is the proof that the rule was rewritten without changing a single record.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_red_light.py src/tvd/rules/red_light.py src/tvd/rules/__init__.py configs/tvd/rules.yaml
git commit -m "Add the red-light rule"
```

### Task 12: Camera files

**Why:** One YAML file per camera, as the spec asks: its frame size, whether it moves, and for each rule a `scene` (where things are in its image) and optional `settings` that override the defaults for this camera only. The camera file only knows the rules by name; each rule checks its own section (**open/closed**: a new rule needs no change here). Two camera files are converted from the notebooks' scene files, and a test checks that they say the same.

**Files:**
- Create: `tests/tvd/test_tvd_camera.py`
- Create: `src/tvd/camera.py`
- Create: `configs/cameras/video-test.yaml`
- Create: `configs/cameras/traffic_video_modified.yaml`

**Interfaces:**
- Consumes: `ComponentConfig`, `StrictModel`, `load_config` (Part 1); `WrongWayScene`, `RedLightScene` in the test (Tasks 9 and 11).
- Produces: `CameraConfig(camera_id, frame_size, motion=fixed, rules)`; `RuleSection(scene, settings={})`; `load_camera(path)`; `load_rule_defaults(path) -> dict[str, dict]`; `effective_settings(defaults, section, rule) -> dict`. `configs/cameras/video-test.yaml` and `configs/cameras/traffic_video_modified.yaml`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_camera.py`:

```python
import pytest
from conftest import ROOT
from notebook_data import load_json, red_light_section, wrong_way_scene

from tvd.camera import RuleSection, effective_settings, load_camera, load_rule_defaults
from tvd.config import ConfigError
from tvd.rules.red_light import RedLightScene
from tvd.rules.wrong_way import WrongWayScene

CAMERAS = ROOT / "configs" / "cameras"
MINIMAL = """
camera_id: gate-1
frame_size: [640, 360]
rules:
  wrong_way:
    scene: {vanishing_point: [0, 0], regions: []}
"""


def write(tmp_path, text: str):
    path = tmp_path / "camera.yaml"
    path.write_text(text)
    return path


def test_the_committed_camera_files_load():
    red, wrong = load_camera(CAMERAS / "traffic_video_modified.yaml"), load_camera(CAMERAS / "video-test.yaml")
    assert (red.frame_size, red.motion.type, list(red.rules)) == ((1920, 1012), "orb", ["red_light"])
    assert red.rules["red_light"].settings == {"entry_offset": 0.0, "pass_margin": 0.2}
    assert (wrong.motion.options["reference_frame"], list(wrong.rules)) == (450, ["wrong_way"])


def test_the_committed_camera_files_say_what_the_notebook_scenes_say():
    scenes = ROOT / "configs" / "scenes"
    wrong, red = load_camera(CAMERAS / "video-test.yaml"), load_camera(CAMERAS / "traffic_video_modified.yaml")
    old_wrong, old_red = load_json(scenes / "video-test.json"), load_json(scenes / "traffic_video_modified.json")
    assert WrongWayScene.model_validate(wrong.rules["wrong_way"].scene) == WrongWayScene.model_validate(wrong_way_scene(old_wrong))
    assert wrong.motion.options["reference_frame"] == old_wrong["coordinates"]["reference_frame_index"]
    section = red_light_section(old_red, "main")
    assert RedLightScene.model_validate(red.rules["red_light"].scene) == RedLightScene.model_validate(section["scene"])
    assert red.rules["red_light"].settings == section["settings"]
    assert (red.motion.options["reference_frame"], red.motion.options["excluded_areas"]) == (old_red["camera"]["reference_frame"], old_red["camera"]["excluded_areas"])


def test_a_camera_without_motion_is_fixed(tmp_path):
    camera = load_camera(write(tmp_path, MINIMAL))
    assert camera.motion.type == "fixed" and camera.rules["wrong_way"].settings == {}


def test_settings_left_empty_mean_none(tmp_path):
    assert load_camera(write(tmp_path, MINIMAL + "    settings:\n")).rules["wrong_way"].settings == {}


def test_a_camera_needs_a_rule(tmp_path):
    with pytest.raises(ConfigError, match="rules: Dictionary should have at least 1 item"):
        load_camera(write(tmp_path, MINIMAL.split("rules:")[0] + "rules: {}\n"))


def test_an_unknown_key_is_named(tmp_path):
    with pytest.raises(ConfigError, match="rules.wrong_way.setings: Extra inputs are not permitted"):
        load_camera(write(tmp_path, MINIMAL + "    setings: {window_seconds: 3}\n"))


def test_a_frame_size_needs_two_numbers(tmp_path):
    with pytest.raises(ConfigError, match="frame_size"):
        load_camera(write(tmp_path, MINIMAL.replace("[640, 360]", "[640]")))


def test_a_camera_changes_only_the_settings_it_names():
    defaults = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
    settings = effective_settings(defaults, RuleSection(scene={}, settings={"window_seconds": 3.0}), "wrong_way")
    assert settings == {**defaults["wrong_way"], "window_seconds": 3.0} and defaults["wrong_way"]["window_seconds"] == 2.0
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_camera.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.camera'`

- [ ] **Step 3: Write the code**

The two camera files are the notebooks' scene files `configs/scenes/video-test.json` and `traffic_video_modified.json`, in the new format. A wrong-way camera lists only the regions that have a direction: the others never mattered to the rule.

`src/tvd/camera.py`:

```python
"""Camera files: what one camera sees, and which rules apply to it. One file per camera, never in code.

The defaults of every rule are in one file for all cameras. A camera file may change any of them for
that camera only, under its rule's ``settings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, RootModel, field_validator

from tvd.config import ComponentConfig, StrictModel, load_config


class RuleSection(StrictModel):
    """One rule of one camera: where things are in its image (``scene``), and its own settings, if any."""

    scene: dict[str, Any]
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("settings", mode="before")
    @classmethod
    def no_settings_when_left_empty(cls, value: Any) -> Any:
        return {} if value is None else value


class CameraConfig(StrictModel):
    camera_id: str
    frame_size: tuple[int, int]  # Width and height. A video of another size is refused.
    motion: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="fixed"))
    rules: dict[str, RuleSection] = Field(min_length=1)


class RuleDefaults(RootModel[dict[str, dict[str, Any]]]):
    """The settings of each rule, by rule name."""


def load_camera(path: str | Path) -> CameraConfig:
    return load_config(path, CameraConfig)


def load_rule_defaults(path: str | Path) -> dict[str, dict[str, Any]]:
    return load_config(path, RuleDefaults).root


def effective_settings(defaults: dict[str, dict[str, Any]], section: RuleSection, rule: str) -> dict[str, Any]:
    """A rule's defaults, with what the camera file changes for this camera. The rule checks the result."""
    return {**defaults.get(rule, {}), **section.settings}
```

`configs/cameras/video-test.yaml`:

```yaml
# A junction filmed from a hand-held camera (data/video-test.mp4). From configs/scenes/video-test.json.
camera_id: video-test
frame_size: [1920, 1080]
motion:                       # the camera moves: positions are compared on frame 450
  type: orb
  options:
    reference_frame: 450
rules:
  wrong_way:
    scene:
      vanishing_point: [930.2, 113.9]   # where the lanes meet; "away_from_camera" points at it
      regions:                          # pixels of the reference frame
        - id: left_carriageway
          polygon: [[733, 247], [930, 250], [931, 380], [556, 369]]
          allowed_direction: toward_camera
          tolerance_deg: 60
        - id: right_carriageway
          polygon: [[934, 250], [1125, 254], [1535, 540], [937, 540]]
          allowed_direction: away_from_camera
          tolerance_deg: 60
```

`configs/cameras/traffic_video_modified.yaml`:

```yaml
# A night street with a signal graphic added to the image (data/traffic_video_modified.mp4, not in git).
# It is a simulation: the graphic is treated as the real signal of the road. From configs/scenes/traffic_video_modified.json.
camera_id: traffic_video_modified
frame_size: [1920, 1012]
motion:                       # the camera moves: positions are compared on frame 357
  type: orb
  options:
    reference_frame: 357
    excluded_areas:           # drawn over the image, so they do not move with the road
      - [1742, 21, 1879, 309]   # the signal graphic
      - [500, 410, 1420, 670]
rules:
  red_light:
    scene:
      stop_line: [[675, 841], [1045, 852], [1864, 888]]   # pixels of the reference frame
      lane_dividers: [675, 1864]
      lane_direction_deg: 169.0
      traffic_direction: toward_camera
      signal_source: simulated_overlay
      signals:
        reader: lamps
        options: {half: 12}
        through:
          - {red: [1810, 79], yellow: [1810, 160], green: [1810, 239]}
        left:
          - {red: [1810, 79], yellow: [1810, 160], green: [1810, 239]}
    settings:                 # vehicles come toward the camera: the box bottom is already the front,
      entry_offset: 0.0       # and the image ends soon after the line
      pass_margin: 0.2
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `134 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_camera.py src/tvd/camera.py configs/cameras/video-test.yaml configs/cameras/traffic_video_modified.yaml
git commit -m "Add camera files"
```

### Task 13: Factory: motion estimators and rules

**Why:** The factory learns two more kinds of component. `build_rules` gives each rule its defaults with the camera's own settings on top, and names the rule in any problem, such as a misspelt setting in a camera file.

**Files:**
- Replace: `tests/tvd/test_tvd_factory.py`
- Replace: `src/tvd/factory.py`

**Interfaces:**
- Consumes: `MOTION_ESTIMATORS` (Task 4), `RULES` (Task 8), `CameraConfig`, `effective_settings` (Task 12), `build` (Part 1).
- Produces: `build_motion_estimator(config: ComponentConfig) -> BaseMotionEstimator`; `build_rules(camera: CameraConfig, defaults) -> list[BaseRule]`.

- [ ] **Step 1: Write the failing test**

`test_tvd_factory.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/test_tvd_factory.py`:

```python
import importlib.util

import pytest
from conftest import ROOT

from tvd.camera import CameraConfig, load_rule_defaults
from tvd.config import ComponentConfig, ConfigError
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker

needs_boxmot = pytest.mark.skipif(importlib.util.find_spec("boxmot") is None, reason="Needs BoxMOT, as on the GPU server.")


def test_a_detector_is_built_by_its_name_with_its_options():
    assert build_detector(ComponentConfig(type="test-moving-car", options={"step": 3})).step == 3


def test_an_unknown_tracker_lists_the_available_ones():
    with pytest.raises(ConfigError, match="Unknown tracker 'deepsort'. Available: botsort, bytetrack, ocsort"):
        build_tracker(ComponentConfig(type="deepsort"))


def test_an_option_the_class_does_not_take_is_reported():
    with pytest.raises(ConfigError, match="test-moving-car: .*unexpected keyword argument 'speed'"):
        build_detector(ComponentConfig(type="test-moving-car", options={"speed": 3}))


@needs_boxmot
def test_a_misspelt_boxmot_option_is_reported():
    with pytest.raises(ConfigError, match="ocsort: .*use_bytes"):
        build_tracker(ComponentConfig(type="ocsort", options={"use_bytes": True}))


CAMERA = {
    "camera_id": "gate-1",
    "frame_size": [640, 360],
    "rules": {
        "wrong_way": {"scene": {"vanishing_point": [320, -1000], "regions": [{"id": "road", "polygon": [[0, 0], [640, 0], [640, 360]], "allowed_direction": "away_from_camera", "tolerance_deg": 60}]}},
        "red_light": {
            "scene": {"stop_line": [[0, 200], [640, 200]], "lane_dividers": [100], "lane_direction_deg": -90, "traffic_direction": "away_from_camera",
                      "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}},
            "settings": {"pass_margin": 0.3},
        },
    },
}


def camera(**changes) -> CameraConfig:
    return CameraConfig.model_validate({**CAMERA, **changes})


def test_rules_are_built_with_the_camera_settings_over_the_defaults():
    wrong_way, red_light = build_rules(camera(), load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml"))
    assert (wrong_way.name, red_light.name, red_light.settings.pass_margin, red_light.settings.entry_offset) == ("wrong_way", "red_light", 0.3, 0.3)


def test_an_unknown_rule_is_named():
    with pytest.raises(ConfigError, match="rules.speeding: Unknown rule 'speeding'. Available: red_light, wrong_way"):
        build_rules(camera(rules={"speeding": {"scene": {}}}), {})


def test_a_misspelt_camera_setting_is_named():
    rules = {"red_light": {**CAMERA["rules"]["red_light"], "settings": {"pass_margn": 0.3}}}
    with pytest.raises(ConfigError, match="rules.red_light: settings.pass_margn: Extra inputs are not permitted"):
        build_rules(camera(rules=rules), load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml"))


def test_the_motion_estimator_is_built_by_its_name():
    assert build_motion_estimator(ComponentConfig(type="orb", options={"reference_frame": 5})).reference_frame == 5
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_factory.py -q
```

Expected: `E   ImportError: cannot import name 'build_motion_estimator' from 'tvd.factory'`

- [ ] **Step 3: Write the code**

`src/tvd/factory.py`:

```python
"""Builds components from their settings: the one place where names in a file become objects.

The pipeline receives its components and never builds them itself (the dependency inversion
principle), so changing any of them is a change of settings, not of code.
"""

from typing import Any, TypeVar

from tvd.camera import CameraConfig, effective_settings
from tvd.config import ComponentConfig, ConfigError
from tvd.detection import DETECTORS, BaseDetector
from tvd.motion import MOTION_ESTIMATORS, BaseMotionEstimator
from tvd.registry import Registry
from tvd.rules import RULES, BaseRule
from tvd.tracking import TRACKERS, BaseTracker

T = TypeVar("T")


def build(registry: Registry[T], config: ComponentConfig) -> T:
    try:
        return registry.create(config.type, **config.options)
    except (ValueError, TypeError) as error:
        raise ConfigError(f"{config.type}: {error}") from error


def build_detector(config: ComponentConfig) -> BaseDetector:
    return build(DETECTORS, config)


def build_tracker(config: ComponentConfig) -> BaseTracker:
    return build(TRACKERS, config)


def build_motion_estimator(config: ComponentConfig) -> BaseMotionEstimator:
    return build(MOTION_ESTIMATORS, config)


def build_rules(camera: CameraConfig, defaults: dict[str, dict[str, Any]]) -> list[BaseRule]:
    """The camera's rules, each with its defaults and the camera's own settings. A problem names the rule."""
    rules = []
    for name, section in camera.rules.items():
        try:
            rules.append(RULES.create(name, scene=section.scene, settings=effective_settings(defaults, section, name)))
        except (ValueError, TypeError) as error:
            raise ConfigError(f"rules.{name}: {error}") from error
    return rules
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `138 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_factory.py src/tvd/factory.py
git commit -m "Build motion estimators and rules from settings"
```

### Task 14: One pass for every rule

**Why:** `check_video` is the one pass the spec asks for: detect and track once, feed every rule's observers (and the detection log, if the camera moves), estimate the motion, and give every rule the same `Observation`. It takes components already built (**dependency inversion**). A video whose size differs from the camera file's is refused at once, because the geometry would be drawn in the wrong place. The test detector can now move the car down the image too. The last test runs the whole path on clip C of `rlr_occasions.mp4` and requires the notebook's crossing records.

**Files:**
- Replace: `tests/tvd/conftest.py`
- Create: `tests/tvd/test_tvd_check.py`
- Create: `src/tvd/check.py`

**Interfaces:**
- Consumes: `track_video`, `DetectionLog` (Task 2); `BaseMotionEstimator` (Task 4); `BaseRule`, `Observation`, `RuleResult` (Task 8); `CameraConfig` (Task 12); the builders (Task 13).
- Produces: `check_video(reader, camera, detector, tracker, track_classes, motion_estimator, rules, start=0, stop=None) -> CheckResult`; `CheckResult(video, camera_id, frames, fps, tracks, seconds, results)` with `as_json()`; `check_frame_size(reader, camera)`. `test-moving-car` takes `step_y`.

- [ ] **Step 1: Write the failing tests**

`conftest.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/conftest.py`:

```python
"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame, TrackedObject
from tvd.signals import SIGNAL_READERS, BaseSignalReader
from tvd.tracking import TRACKERS, BaseTracker

ROOT = Path(__file__).resolve().parents[2]


@DETECTORS.register("test-moving-car")
class MovingCarDetector(BaseDetector):
    """A car that moves by ``step`` px to the right and ``step_y`` px down per frame, and a person who stands still."""

    def __init__(self, step: float = 5.0, step_y: float = 0.0) -> None:
        self.step = step
        self.step_y = step_y

    def detect(self, frame: Frame) -> list[Detection]:
        x, y = 10 + self.step * frame.index, 50 + self.step_y * frame.index
        return [Detection(Box(x, y, x + 40, y + 30), 2, "car", 0.9), Detection(Box(0, 0, 10, 30), 0, "person", 0.8)]


@TRACKERS.register("test-same-id")
class SameIdTracker(BaseTracker):
    """Gives every detection track id 1: enough to test what surrounds a tracker."""

    def __init__(self) -> None:
        self.resets = 0

    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        return [TrackedObject(1, d.box, d.class_id, d.class_name, d.confidence) for d in detections]

    def reset(self) -> None:
        self.resets += 1


@SIGNAL_READERS.register("test-fixed-state")
class FixedStateReader(BaseSignalReader):
    """A signal head that always shows ``state``."""

    def __init__(self, state: str = "red") -> None:
        self.state = state

    def read(self, image) -> str:
        return self.state


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
```

`tests/tvd/test_tvd_check.py`:

```python
import importlib.util

import pytest
from conftest import ROOT
from notebook_data import RED_LIGHT_SCENE, RLR_VIDEO, load_json, red_light_section

from tvd.camera import CameraConfig, load_rule_defaults
from tvd.check import check_video
from tvd.config import ComponentConfig, ConfigError
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
from tvd.video import VideoReader

DEFAULTS = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
# The test car comes down the image by 1 px per frame: towards the camera, against the only allowed direction.
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
needs_gpu_and_rlr = pytest.mark.skipif(
    any(importlib.util.find_spec(n) is None for n in ("ultralytics", "boxmot")) or not RLR_VIDEO.is_file() or not (ROOT / "weights" / "yolo26m.pt").is_file(),
    reason="Needs Ultralytics, BoxMOT, the weights and data/raw/rlr_occasions.mp4, as on the GPU server.",
)


def run(sample_video, camera: dict = CAMERA):
    config = CameraConfig.model_validate(camera)
    with VideoReader(sample_video) as reader:
        return check_video(
            reader, config, build_detector(ComponentConfig(type="test-moving-car", options={"step": 0, "step_y": 1})),
            build_tracker(ComponentConfig(type="test-same-id")), {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS),
        )


def test_one_pass_gives_every_rule_the_same_tracks(sample_video):
    result = run(sample_video)
    assert (result.frames, result.tracks, list(result.results)) == (240, 1, ["red_light", "wrong_way"])
    (red,), (wrong,) = result.results["red_light"].findings, result.results["wrong_way"].findings
    assert (red.trigger_frame, red.evidence["signal_state"]) == (70, "red")
    assert (wrong.start_frame, wrong.trigger_frame, wrong.end_frame) == (0, 48, 239)


def test_the_result_can_be_written_as_json(sample_video):
    import json

    data = json.loads(json.dumps(run(sample_video).as_json()))
    assert data["rules"]["red_light"]["verdicts"] == {"candidate": 1} and data["rules"]["wrong_way"]["settings"]["window_seconds"] == 2.0


def test_a_video_of_another_size_is_refused(sample_video):
    with pytest.raises(ConfigError, match="Camera test-camera has 1920x1080 frames, but traffic_sample.mp4 is 640x360"):
        run(sample_video, {**CAMERA, "frame_size": [1920, 1080]})


@needs_gpu_and_rlr
def test_the_same_crossings_as_the_notebooks_from_the_video(monkeypatch):
    from traffic_violation_detection import red_light as notebook
    from traffic_violation_detection.detection import load_detector

    monkeypatch.chdir(ROOT)
    scene, clip_name = load_json(RED_LIGHT_SCENE), "C"
    clip = scene["clips"][clip_name]
    tracks, states, _ = notebook.track_and_read_clip(RLR_VIDEO, clip, load_detector())
    expected = notebook.evaluate_clip(notebook.join_clip_tracks(tracks, None), clip, states, scene["lane_direction_deg"], 1280, 720, 30.0)

    camera = CameraConfig.model_validate({"camera_id": "rlr-C", "frame_size": [1280, 720], "rules": {"red_light": red_light_section(scene, clip_name)}})
    perception = {"weights": "yolo26m.pt", "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]}
    with VideoReader(RLR_VIDEO) as reader:
        result = check_video(
            reader, camera, build_detector(ComponentConfig(type="yolo", options=perception)), build_tracker(ComponentConfig(type="ocsort", options={"use_byte": True})),
            {"bicycle", "car", "motorcycle", "bus", "truck"}, build_motion_estimator(camera.motion), build_rules(camera, DEFAULTS), clip["start"], clip["end"],
        )
    assert result.results["red_light"].records == expected and len(result.results["red_light"].findings) == 2
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_check.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.check'`

- [ ] **Step 3: Write the code**

`src/tvd/check.py`:

```python
"""One pass over a video for one camera: detection and tracking, every rule's per-frame work, the camera's motion, then the rules."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from tvd.camera import CameraConfig
from tvd.config import ConfigError
from tvd.detection import BaseDetector
from tvd.motion import BaseMotionEstimator
from tvd.observers import DetectionLog
from tvd.perception import track_video
from tvd.rules import BaseRule, Observation, RuleResult
from tvd.tracking import BaseTracker
from tvd.video import VideoReader


@dataclass
class CheckResult:
    video: str
    camera_id: str
    frames: int
    fps: float
    tracks: int
    seconds: float
    results: dict[str, RuleResult]

    def as_json(self) -> dict[str, Any]:
        return {
            "video": self.video, "camera_id": self.camera_id, "frames": self.frames, "fps": self.fps, "tracks": self.tracks, "seconds": self.seconds,
            "rules": {
                name: {"settings": r.settings, "verdicts": r.verdict_counts(), "findings": [asdict(f) for f in r.findings], "records": r.records}
                for name, r in self.results.items()
            },
        }


def check_frame_size(reader: VideoReader, camera: CameraConfig) -> None:
    """A camera file's geometry is drawn for one frame size: a video of another size is refused before any work."""
    if (reader.width, reader.height) != tuple(camera.frame_size):
        width, height = camera.frame_size
        raise ConfigError(f"Camera {camera.camera_id} has {width}x{height} frames, but {reader.path.name} is {reader.width}x{reader.height}")


def check_video(
    reader: VideoReader,
    camera: CameraConfig,
    detector: BaseDetector,
    tracker: BaseTracker,
    track_classes: set[str],
    motion_estimator: BaseMotionEstimator,
    rules: list[BaseRule],
    start: int = 0,
    stop: int | None = None,
) -> CheckResult:
    """Detect and track once, and give every rule the same tracks. Build new rules for each video: their observers keep what they saw."""
    check_frame_size(reader, camera)
    began = time.perf_counter()
    log = DetectionLog() if motion_estimator.needs_detections else None
    observers = [o for rule in rules for o in rule.frame_observers()] + ([log] if log else [])
    history = track_video(reader, detector, tracker, track_classes, start, stop, observers)
    end = start + history.frames_seen
    motion = motion_estimator.estimate(reader, log.by_frame if log else {}, start, end)
    observation = Observation(history.tracks, motion, reader.fps, reader.width, reader.height, start, end)
    results = {rule.name: rule.evaluate(observation) for rule in rules}
    return CheckResult(reader.path.name, camera.camera_id, history.frames_seen, reader.fps, len(history.tracks), round(time.perf_counter() - began, 1), results)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `142 passed`.

`test_the_same_crossings_as_the_notebooks_from_the_video` is the whole Phase 1 chain on clip C, from pixels to records: YOLO, OC-SORT, the night signal reader, the joins and the rule. It takes about 15 s.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/conftest.py tests/tvd/test_tvd_check.py src/tvd/check.py
git commit -m "Apply every rule of a camera in one pass"
```

### Task 15: The check command

**Why:** The command line is the composition root again: it reads every settings file and opens the video before any model loads, so a mistake costs a fraction of a second. Then it builds the components and runs `check_video`, and writes every finding, every record and the settings used to one JSON file.

**Files:**
- Replace: `tests/tvd/test_tvd_cli.py`
- Replace: `src/tvd/cli.py`

**Interfaces:**
- Consumes: `load_camera`, `load_rule_defaults` (Task 12); `build_*` (Task 13); `check_video`, `check_frame_size` (Task 14).
- Produces: `python -m tvd.cli check --video <file> --camera <yaml> --output <json> [--perception <yaml>] [--rules <yaml>]`; `run_check(...) -> dict`; `check_summary(data) -> str`.

- [ ] **Step 1: Write the failing test**

`test_tvd_cli.py` is replaced as a whole: the new version keeps every earlier test and adds this task's.

`tests/tvd/test_tvd_cli.py`:

```python
import json

from tvd.cli import main

SETTINGS = """
detector:
  type: test-moving-car
tracker:
  type: test-same-id
track_classes: [car]
"""


def test_track_writes_the_tracks_and_prints_a_summary(tmp_path, capsys, sample_video):
    settings = tmp_path / "perception.yaml"
    settings.write_text(SETTINGS)
    output = tmp_path / "out" / "tracks.json"
    assert main(["track", "--video", str(sample_video), "--config", str(settings), "--output", str(output)]) == 0
    assert "traffic_sample.mp4: 240 frames, 1 tracks, detector test-moving-car, tracker test-same-id" in capsys.readouterr().out
    data = json.loads(output.read_text())
    assert (data["frames"], data["tracks"]) == (240, 1)
    assert data["track_list"][0]["points"][0] == [0, 10.0, 50.0, 50.0, 80.0, 0.9]


def test_a_bad_settings_file_is_reported_without_a_traceback(tmp_path, capsys, sample_video):
    settings = tmp_path / "perception.yaml"
    settings.write_text(SETTINGS.replace("test-same-id", "deepsort"))
    assert main(["track", "--video", str(sample_video), "--config", str(settings), "--output", str(tmp_path / "t.json")]) == 2
    assert "error: deepsort: Unknown tracker 'deepsort'" in capsys.readouterr().err


CAMERA_FILE = """
camera_id: test-camera
frame_size: [640, 360]
rules:
  red_light:
    scene:
      stop_line: [[0, 150], [640, 150]]
      lane_dividers: [0, 600]
      lane_direction_deg: 90
      traffic_direction: toward_camera
      signals: {reader: test-fixed-state, through: [{state: red}]}
    settings: {entry_offset: 0.0, pass_margin: 0.2}
"""
DOWN = SETTINGS.replace("type: test-moving-car", "type: test-moving-car\n  options: {step: 0, step_y: 1}")


def test_check_writes_the_findings_and_prints_a_summary(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE)
    (tmp_path / "perception.yaml").write_text(DOWN)
    output = tmp_path / "findings.json"
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(output), "--perception", str(tmp_path / "perception.yaml")]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "traffic_sample.mp4: 240 frames, 1 tracks, camera test-camera" in printed and "red_light: 1 candidates for review; candidate 1" in printed
    assert json.loads(output.read_text())["rules"]["red_light"]["findings"][0]["trigger_frame"] == 70


def test_check_refuses_a_video_of_another_camera(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE.replace("[640, 360]", "[1280, 720]"))
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(tmp_path / "f.json")]
    assert main(argv) == 2
    assert "error: Camera test-camera has 1280x720 frames, but traffic_sample.mp4 is 640x360" in capsys.readouterr().err
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_cli.py -q
```

Expected: `2 failed, 2 passed`: the two new tests stop with `SystemExit: 2`, after `tvd: error: argument command: invalid choice: 'check' (choose from 'track')`

- [ ] **Step 3: Write the code**

`src/tvd/cli.py`:

```python
"""Command line.

- ``python -m tvd.cli track --video <file> --config <yaml> --output <json>``: detect and track.
- ``python -m tvd.cli check --video <file> --camera <yaml> --output <json>``: apply the camera's rules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tvd.camera import load_camera, load_rule_defaults
from tvd.check import check_frame_size, check_video
from tvd.config import ConfigError, PerceptionConfig, load_config
from tvd.factory import build_detector, build_motion_estimator, build_rules, build_tracker
from tvd.perception import track_video
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader

PERCEPTION = Path("configs/tvd/perception.yaml")
RULE_DEFAULTS = Path("configs/tvd/rules.yaml")


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
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            summary = run_track(args.video, args.config, args.output)
            print(f"{summary['video']}: {summary['frames']} frames, {summary['tracks']} tracks, detector {summary['detector']}, tracker {summary['tracker']}, {summary['seconds']} s")
        else:
            print(check_summary(run_check(args.video, args.camera, args.output, args.perception, args.rules)))
    except (ConfigError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `144 passed`.

- [ ] **Step 5: Check the red-light camera, which moves**

```bash
scripts/server.sh run 'python -m tvd.cli check --video data/traffic_video_modified.mp4 --camera configs/cameras/traffic_video_modified.yaml --output outputs/tvd/traffic_video_modified_findings.json'
```

Expected, after some log lines from BoxMOT (the time varies; ORB takes most of it):

```text
traffic_video_modified.mp4: 715 frames, 45 tracks, camera traffic_video_modified, 107.2 s
  red_light: 2 candidates for review; candidate 2, not_evaluable_signal 1, signal_not_red 7, stopped_at_line 1
```

These are the outcomes of Notebook 08, in `outputs/08_end_to_end_baseline/traffic_video_modified/traffic_video_modified_summary.json`. The JSON file holds `settings`, `verdicts`, `findings` and the 11 `records` of the rule. Its first finding:

```text
"rule": "red_light", "track_id": "23", "class_name": "car", "start_frame": 260, "trigger_frame": 317, "end_frame": 550,
"why": "Entered in lane 'through' while the through signal was red (1.0 s after it was first seen in that state), and continued past the stop line.",
"evidence": {"x": 1481.6, "lane": "through", "signal_state": "red", "movement": "straight", ..., "signal_source": "simulated_overlay"}
```

- [ ] **Step 6: Check the wrong-way camera, which moves too**

```bash
scripts/server.sh run 'python -m tvd.cli check --video data/video-test.mp4 --camera configs/cameras/video-test.yaml --output outputs/tvd/video-test_findings.json'
```

Expected, in about 3.5 minutes:

```text
video-test.mp4: 900 frames, 136 tracks, camera video-test, 202.3 s
  wrong_way: 0 candidates for review; no_violation 20, not_evaluable 11, not_in_directional_region 105
```

The same counts as Notebook 05 (`status_counts` in `outputs/05_violation_detection/video-test_events.json`): nobody drives the wrong way in this video, which Task 9 checked by playing it backwards.

- [ ] **Step 7: Change a setting for one camera only**

In `configs/cameras/traffic_video_modified.yaml`, add a line under `settings`:

```yaml
    settings:
      entry_offset: 0.0
      pass_margin: 0.2
      min_track_frames: 60    # only judge tracks seen for at least 60 frames
```

```bash
scripts/server.sh sync && scripts/server.sh run 'python -m tvd.cli check --video data/traffic_video_modified.mp4 --camera configs/cameras/traffic_video_modified.yaml --output outputs/tvd/traffic_video_modified_findings.json'
```

Expected: `signal_not_red 6` instead of 7, and the rest the same: one short track is no longer judged. The JSON file's `settings` shows `"min_track_frames": 60`. No code changed, and no other camera is affected. Remove the line again before the next step.

- [ ] **Step 8: See how mistakes are reported**

The camera file of another camera (`--camera configs/cameras/video-test.yaml` with `traffic_video_modified.mp4`), refused in about 0.5 s, before any model loads:

```text
error: Camera video-test has 1920x1080 frames, but traffic_video_modified.mp4 is 1920x1012
```

A misspelt setting in a camera file (`pass_margn: 0.2`):

```text
error: rules.red_light: settings.pass_margn: Extra inputs are not permitted
```

Each exits with code 2.

- [ ] **Step 9: Commit**

```bash
git add tests/tvd/test_tvd_cli.py src/tvd/cli.py
git commit -m "Add the check command"
```

---

## When Part 2 is done

- [ ] `scripts/server.sh test tests/tvd -q` gives `144 passed`, and `scripts/server.sh test -q` still passes the notebooks' tests too.
- [ ] `check` gives the outcomes of Notebook 08 on `traffic_video_modified.mp4` and of Notebook 05 on `video-test.mp4`.
- [ ] Changing a setting for one camera changes that camera's results, and no code.
- [ ] You can say, for each of the five SOLID principles, where it is in `src/tvd/rules/` and `src/tvd/signals/`.

**Exercises, if you want them** (not checked while writing the plan):

- A **third signal reader**: one that averages the colour of a whole head by day. It is a new class with `@SIGNAL_READERS.register("average")`, and a camera file chooses it with `reader: average`. Nothing else changes.
- A **speed rule**: a new `BaseRule` that measures how far a track moves in reference pixels per second. It needs a scale (metres per pixel) in its scene, and a limit in `configs/tvd/rules.yaml`.

**Next:** Part 3, plates and evidence, turns each finding into a reviewable record: a clip, three images with the stop line and the signal state drawn on them, the plate read in a worker process, and the record of schema version 2. Its plan is written once Part 2 is done.
