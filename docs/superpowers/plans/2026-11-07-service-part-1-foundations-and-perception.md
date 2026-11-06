# Service Part 1: Foundations and Perception — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first layer of the review service as replaceable components: data types, a registry, a video reader, a detector and three trackers behind base classes, settings files, and a `track` command. Changing the tracker is a one-word change in a YAML file.

**Architecture:** A new package, `src/tvd/`, next to the notebooks' package `traffic_violation_detection`, which stays as it is. Each kind of component has an abstract base class and a `Registry`. Concrete classes register under a name, a factory builds the one a settings file names, and the pipeline only knows the base classes. Tests check the new code against the notebooks' results.

**Tech Stack:** Python 3.11, OpenCV 5.0.0, NumPy 2.4.6, Ultralytics 8.4.197 (YOLO26m), BoxMOT 26.0.0, pydantic 2, PyYAML 6, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-review-service-design.md`

**Verified:** every file below, and the state after every task, was run on ai_server_2 (NVIDIA L40S) before this plan was written. After the last task: 54 tests pass, and `track` gives 136 tracks on `video-test.mp4` with OC-SORT, as Notebook 03 does.

## The five parts

| Part | Builds | Plan |
|---|---|---|
| **1. Foundations and perception** | Data types, registry, video reader, detector, trackers, settings, `track` command | This document |
| 2. Scene and rules | Camera files, signal reading, the wrong-way and red-light rules rewritten | Written after Part 1 |
| 3. Plates and evidence | Plate detection and OCR in a worker process, voting, clips, images with the stop line and signal state, records | Later |
| 4. The service | One pass for both rules, job queue, storage, REST API with FastAPI | Later |
| 5. Deployment and MLOps | Docker, running on ai_server_2, logging, health checks, model versions | Later |

Each part ends with working, tested software. Each later plan is written from what the earlier parts actually became.

## Global Constraints

- Python 3.11, in the `tvd` environment: on ai_server_2, `~/workspace/dungvu/envs/tvd`.
- Never let pip change OpenCV (5.0.0.93) or NumPy (2.4.6). BoxMOT is installed with `--no-deps`.
- New dependencies: `pydantic>=2.6,<3` and `pyyaml>=6,<7`, in a new `service` extra.
- New code goes in `src/tvd/`, and its tests in `tests/tvd/`. Test files are named `test_tvd_*.py`, so they never clash with the existing `tests/test_*.py`.
- Tests that need the GPU, Ultralytics, BoxMOT or `weights/yolo26m.pt` skip themselves where those are missing. The full count below is from the GPU server.
- Notebook results are the reference. Where a test compares with them, the new code must match **exactly**.

## Review Focus

- **`options:` left empty in YAML** reads as null. It must mean "no options", not an error. Pinned in Task 10, `test_options_left_empty_mean_no_options`.
- **A wrong video path** must fail before any model loads, in well under a second. Task 12 opens the video first. Checked by hand in Task 12, Step 7 (0.3 s on the server).
- **A misspelt setting or tracker name** must name the culprit. Pinned in Tasks 10 and 11.
- **A frame with no detections** must give no tracked objects, not crash BoxMOT's arrays. Pinned in Task 7, `test_no_detections_give_no_tracked_objects`.
- **Running two videos with one tracker object** must not carry tracks over. `track_video` resets the tracker first, pinned in Task 9.

## How you will work

You edit code on your laptop, in this repository, and run the tests on ai_server_2, where the GPU and the models are. `scripts/server.sh` (Task 0) does the copying and the remote commands:

| Command | Does |
|---|---|
| `scripts/server.sh sync` | Copies `src/`, `tests/`, `configs/`, `scripts/` and `pyproject.toml` to the server |
| `scripts/server.sh test tests/tvd -q` | Runs pytest there |
| `scripts/server.sh run <command>` | Runs any command there, in the project folder, with the `tvd` environment |

The server's copy is already installed in editable mode from its `src/` folder, so a new package there is importable as soon as it is copied. Tests that need no GPU also run on your laptop with `python -m pytest tests/tvd -q`; the GPU ones skip there.

## File structure

```text
src/tvd/
├── __init__.py
├── domain.py                 Task 1   Box, Detection, TrackedObject, Frame
├── registry.py               Task 2   Registry: names -> classes
├── video.py                  Task 3   VideoReader
├── detection/
│   ├── __init__.py           Task 4/5 exports; importing it registers the detectors
│   ├── base.py               Task 4   BaseDetector, DETECTORS
│   └── yolo.py               Task 5   YoloDetector  ("yolo")
├── tracking/
│   ├── __init__.py           Task 6/7 exports; importing it registers the trackers
│   ├── base.py               Task 6   BaseTracker, TRACKERS
│   ├── boxmot_trackers.py    Task 7   BoxmotTracker, ByteTrackTracker, OcSortTracker, BotSortTracker
│   └── history.py            Task 8   TrackPoint, Track, TrackHistory
├── perception.py             Task 9   track_video
├── config.py                 Task 10  ConfigError, ComponentConfig, PerceptionConfig, load_config
├── factory.py                Task 11  build_detector, build_tracker
└── cli.py                    Task 12  python -m tvd.cli track ...
configs/tvd/perception.yaml   Task 10
scripts/server.sh             Task 0
tests/tvd/conftest.py         Task 3/4/6  test doubles: test-moving-car, test-same-id
tests/tvd/test_tvd_*.py       one per task
```

**How the parts depend on each other.** Arrows point at what a module uses. Only `factory.py` and `cli.py` know concrete classes, by name.

```text
cli ──> factory ──> registry <── detection/yolo, tracking/boxmot_trackers
 │         └──> config
 └──> perception ──> detection/base, tracking/base, tracking/history, video ──> domain
```

## SOLID in this part

| Principle | Where |
|---|---|
| **S**ingle responsibility | One job per module: `video.py` reads, `history.py` collects, `config.py` checks settings, `factory.py` builds |
| **O**pen/closed | A new detector or tracker is a new class with `@DETECTORS.register` or `@TRACKERS.register`; nothing else changes |
| **L**iskov substitution | Every `BaseTracker` behaves the same to its caller; the test doubles run in the same tests as the real classes |
| **I**nterface segregation | `BaseDetector` has one method, `BaseTracker` two: only what the pipeline needs |
| **D**ependency inversion | `track_video` takes `BaseDetector` and `BaseTracker`; only the factory and the command line build concrete classes |

---

### Task 0: Workspace and tools

**Why:** a branch for this part, the two new dependencies, and the script that runs your code on the GPU server.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tvd/__init__.py`, `scripts/server.sh`

- [ ] **Step 1: Create a branch**

```bash
git checkout -b service/part-1
```

- [ ] **Step 2: Add the `service` extra to `pyproject.toml`**

Under `[project.optional-dependencies]`, next to the other extras:

```toml
# For the review service (src/tvd). pydantic (MIT) checks settings files; PyYAML (MIT) reads them.
service = [
    "pydantic>=2.6,<3",
    "pyyaml>=6.0,<7",
]
```

- [ ] **Step 3: Create the package**

`src/tvd/__init__.py`:

```python
"""The violation review service: the Phase 1 baseline rebuilt from replaceable components."""
```

- [ ] **Step 4: Create the server script, and make it executable**

`scripts/server.sh`:

```bash
#!/usr/bin/env bash
# Work on the GPU server from this machine.
#   scripts/server.sh sync             copy src/, tests/, configs/, scripts/ and pyproject.toml to the server
#   scripts/server.sh test [args]      run pytest there, e.g. scripts/server.sh test tests/tvd -q
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
        rsync -az --timeout=60 -e "${SSH[*]}" --exclude '__pycache__/' src tests configs scripts pyproject.toml "$HOST:$REMOTE_DIR/"
        echo "synced to $HOST:$REMOTE_DIR"
        ;;
    test)
        shift
        remote python -m pytest "$@"
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

```bash
chmod +x scripts/server.sh
```

The server's address and folders can be changed with the variables `TVD_SERVER`, `TVD_REMOTE_DIR` and `TVD_REMOTE_ENV`. The connection through the VPN sometimes drops for a minute; if a command times out, run it again.

- [ ] **Step 5: Install the new dependencies, on your laptop and on the server**

```bash
python -m pip install "pydantic>=2.6,<3" "pyyaml>=6,<7"
scripts/server.sh sync
scripts/server.sh run 'python -m pip install "pydantic>=2.6,<3" "pyyaml>=6,<7"'
```

Expected on the server: `Requirement already satisfied` for both. pydantic was installed there while this plan was checked.

- [ ] **Step 6: Check that the server sees the new package**

```bash
scripts/server.sh run 'python -c "import tvd, pydantic; print(tvd.__doc__); print(pydantic.VERSION)"'
```

Expected:

```text
The violation review service: the Phase 1 baseline rebuilt from replaceable components.
2.14.3
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/tvd/__init__.py scripts/server.sh
git commit -m "Start the service package and add the server script"
```

### Task 1: Domain types

**Why:** Every component passes the same small, frozen data objects to the next. **Single responsibility:** this file only describes data, and imports no model and no tracking library. `frozen=True` means no component can change data another one holds.

**Files:**
- Create: `tests/tvd/test_tvd_domain.py`
- Create: `src/tvd/domain.py`

**Interfaces:**
- Consumes: Nothing.
- Produces: `Box(x1, y1, x2, y2)` with `width`, `height`, `area`, `center`, `iou(other) -> float`, `as_list() -> list[float]`; `Detection(box, class_id, class_name, confidence)`; `TrackedObject(track_id, box, class_id, class_name, confidence)`; `Frame(index, timestamp_s, image)`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_domain.py`:

```python
import numpy as np
import pytest

from tvd.domain import Box, Frame


def test_box_measures():
    box = Box(10, 20, 50, 80)
    assert (box.width, box.height, box.area, box.center) == (40, 60, 2400, (30.0, 50.0))


def test_a_box_with_reversed_corners_is_refused():
    with pytest.raises(ValueError, match="reversed"):
        Box(50, 20, 10, 80)


@pytest.mark.parametrize("other, expected", [(Box(0, 0, 10, 10), 1.0), (Box(5, 0, 15, 10), 1 / 3), (Box(20, 20, 30, 30), 0.0)])
def test_iou(other, expected):
    assert Box(0, 0, 10, 10).iou(other) == pytest.approx(expected)


def test_a_box_with_no_area_overlaps_nothing():
    assert Box(5, 5, 5, 5).iou(Box(5, 5, 5, 5)) == 0.0


def test_frames_are_never_equal_by_value():
    image = np.zeros((2, 2, 3), np.uint8)
    assert Frame(0, 0.0, image) != Frame(0, 0.0, image)
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_domain.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.domain'`

- [ ] **Step 3: Write the code**

`src/tvd/domain.py`:

```python
"""Plain data shared by every component. Nothing here depends on a model or a tracking library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in pixels, from its top-left corner (x1, y1) to its bottom-right corner (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Box corners are reversed: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: Box) -> float:
        """Intersection over union: 1 for the same box, 0 for boxes that do not touch."""
        width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class Detection:
    """One object found in one frame by a detector."""

    box: Box
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class TrackedObject:
    """One object in one frame, with the identity a tracker gave it across frames."""

    track_id: int
    box: Box
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True, eq=False)
class Frame:
    """One decoded video frame. ``eq=False``: comparing two images pixel by pixel is never what a test means."""

    index: int
    timestamp_s: float
    image: np.ndarray
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_domain.py src/tvd/domain.py
git commit -m "Add the service's domain types"
```

### Task 2: Registry

**Why:** A settings file names an implementation (`ocsort`, `bytetrack`), and the registry turns that name into a class. **Open/closed:** a new tracker registers itself, and no existing code changes.

**Files:**
- Create: `tests/tvd/test_tvd_registry.py`
- Create: `src/tvd/registry.py`

**Interfaces:**
- Consumes: Nothing.
- Produces: `Registry[T](kind)` with `register(name)` (a class decorator), `create(name, **options) -> T`, `names() -> list[str]`. Unknown names and duplicate names raise `ValueError`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_registry.py`:

```python
import pytest

from tvd.registry import Registry


class Shape:
    pass


def shapes_with_a_square():
    shapes: Registry[Shape] = Registry("shape")

    @shapes.register("square")
    class Square(Shape):
        def __init__(self, side: float = 1.0) -> None:
            self.side = side

    return shapes, Square


def test_create_builds_the_registered_class_with_its_options():
    shapes, Square = shapes_with_a_square()
    square = shapes.create("square", side=3)
    assert isinstance(square, Square) and square.side == 3


def test_a_name_can_be_registered_only_once():
    shapes, _ = shapes_with_a_square()
    with pytest.raises(ValueError, match="already registered"):

        @shapes.register("square")
        class Other(Shape):
            pass


def test_an_unknown_name_lists_the_known_ones():
    shapes, _ = shapes_with_a_square()
    with pytest.raises(ValueError, match="Unknown shape 'circle'. Available: square"):
        shapes.create("circle")


def test_names_are_sorted():
    shapes, _ = shapes_with_a_square()
    shapes.register("circle")(type("Circle", (Shape,), {}))
    assert shapes.names() == ["circle", "square"]
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_registry.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.registry'`

- [ ] **Step 3: Write the code**

`src/tvd/registry.py`:

```python
"""A table from names to classes, so that a settings file can choose an implementation by its name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Implementations add themselves with ``@REGISTRY.register("name")``, and ``create`` builds one by its name.

    Adding an implementation never changes the code that uses it (the open/closed principle).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def add(cls: type[T]) -> type[T]:
            if name in self._classes:
                raise ValueError(f"A {self._kind} named {name!r} is already registered: {self._classes[name].__name__}")
            self._classes[name] = cls
            return cls

        return add

    def create(self, name: str, **options: Any) -> T:
        if name not in self._classes:
            raise ValueError(f"Unknown {self._kind} {name!r}. Available: {', '.join(self.names()) or 'none'}")
        return self._classes[name](**options)

    def names(self) -> list[str]:
        return sorted(self._classes)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_registry.py src/tvd/registry.py
git commit -m "Add a registry of implementations by name"
```

### Task 3: Video reader

**Why:** Reading frames is one job (**single responsibility**), done once for every component. It decodes from the first frame so that frame indexes are always exact, which every rule later relies on. `conftest.py` starts here with the path to the sample video; later tasks add test doubles to it.

**Files:**
- Create: `tests/tvd/conftest.py`
- Create: `tests/tvd/test_tvd_video.py`
- Create: `src/tvd/video.py`

**Interfaces:**
- Consumes: `Frame` (Task 1).
- Produces: `VideoReader(path)` with `fps`, `width`, `height`, `frame_count`, `frames(start=0, stop=None) -> Iterator[Frame]`, `close()`, and `with` support. A missing file raises `FileNotFoundError`; a file OpenCV cannot open raises `RuntimeError`.

- [ ] **Step 1: Write the failing tests**

`tests/tvd/conftest.py`:

```python
"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
```

`tests/tvd/test_tvd_video.py`:

```python
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
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_video.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.video'`

- [ ] **Step 3: Write the code**

`src/tvd/video.py`:

```python
"""Reading video files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from tvd.domain import Frame


class VideoReader:
    """The frames of a video file, in order. Use it in a ``with`` block, so that the file is closed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"OpenCV could not open the video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            self.close()
            raise RuntimeError(f"The video reports no frame rate: {self.path}")

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
        """Frames ``start`` to ``stop`` (not included), decoded from the first frame, so that every index is exact."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while stop is None or index < stop:
            if not self._capture.grab():
                return
            if index >= start:
                ok, image = self._capture.retrieve()
                if not ok:
                    return
                yield Frame(index, index / self.fps, image)
            index += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `17 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/conftest.py tests/tvd/test_tvd_video.py src/tvd/video.py
git commit -m "Add the video reader"
```

### Task 4: Detector interface

**Why:** `BaseDetector` is the contract every detector keeps. The pipeline will depend on it, never on YOLO (**dependency inversion**). The test double `MovingCarDetector` in `conftest.py` is a second detector, and it works anywhere a real one does (**Liskov substitution**). That is also how the pipeline gets tested without a GPU.

**Files:**
- Replace: `tests/tvd/conftest.py`
- Create: `tests/tvd/test_tvd_detection.py`
- Create: `src/tvd/detection/base.py`
- Create: `src/tvd/detection/__init__.py`

**Interfaces:**
- Consumes: `Registry` (Task 2), `Detection`, `Frame` (Task 1).
- Produces: `BaseDetector.detect(frame: Frame) -> list[Detection]` (abstract), and `DETECTORS: Registry[BaseDetector]`. Test double `test-moving-car`.

- [ ] **Step 1: Write the failing tests**

`conftest.py` is replaced as a whole: it keeps what it had, and gains a test double.

`tests/tvd/conftest.py`:

```python
"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame

ROOT = Path(__file__).resolve().parents[2]


@DETECTORS.register("test-moving-car")
class MovingCarDetector(BaseDetector):
    """A car that moves right by ``step`` px per frame, and a person who stands still."""

    def __init__(self, step: float = 5.0) -> None:
        self.step = step

    def detect(self, frame: Frame) -> list[Detection]:
        x = 10 + self.step * frame.index
        return [Detection(Box(x, 50, x + 40, 80), 2, "car", 0.9), Detection(Box(0, 0, 10, 30), 0, "person", 0.8)]


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
```

`tests/tvd/test_tvd_detection.py`:

```python
import numpy as np
import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Frame


def test_a_detector_must_implement_detect():
    class Incomplete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_the_test_double_finds_a_moving_car():
    car = DETECTORS.create("test-moving-car", step=2).detect(Frame(3, 0.1, np.zeros((1, 1, 3), np.uint8)))[0]
    assert car.class_name == "car" and car.box.x1 == 16
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_detection.py -q
```

Expected: `ImportError while loading conftest ... ModuleNotFoundError: No module named 'tvd.detection'`

- [ ] **Step 3: Write the code**

`src/tvd/detection/base.py`:

```python
"""What every detector offers."""

from abc import ABC, abstractmethod

from tvd.domain import Detection, Frame
from tvd.registry import Registry


class BaseDetector(ABC):
    """Finds objects in one frame.

    Every detector returns the same ``Detection`` type, so nothing after it depends on a model library.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """The objects found in ``frame``."""


DETECTORS: Registry[BaseDetector] = Registry("detector")
```

`src/tvd/detection/__init__.py`:

```python
"""Detectors. Importing this package registers every detector in it."""

from tvd.detection.base import DETECTORS, BaseDetector

__all__ = ["DETECTORS", "BaseDetector"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `19 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/conftest.py tests/tvd/test_tvd_detection.py src/tvd/detection/base.py src/tvd/detection/__init__.py
git commit -m "Add the detector interface"
```

### Task 5: YOLO detector

**Why:** `YoloDetector` adapts Ultralytics to `BaseDetector` (the **adapter** pattern). Ultralytics is imported only when a YOLO detector is built, so the rest of the package works without it. The last test is the most important one in this part: the new detector must give **exactly** the boxes of the notebooks, which proves the rewrite changed nothing.

**Files:**
- Replace: `tests/tvd/test_tvd_detection.py`
- Create: `src/tvd/detection/yolo.py`
- Replace: `src/tvd/detection/__init__.py`

**Interfaces:**
- Consumes: `BaseDetector`, `DETECTORS` (Task 4); `VideoReader` (Task 3) in the test.
- Produces: `YoloDetector(weights, image_size=1280, confidence=0.10, iou=0.7, classes=None, weights_dir="weights")`, registered as `"yolo"`. Boxes are rounded to 0.01 px and confidences to 4 decimals, as in the notebooks.

- [ ] **Step 1: Write the failing test**

This replaces the test file of the previous task with its complete version.

`tests/tvd/test_tvd_detection.py`:

```python
import importlib.util

import numpy as np
import pytest
from conftest import ROOT

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Frame
from tvd.video import VideoReader

needs_yolo = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not (ROOT / "weights" / "yolo26m.pt").is_file(),
    reason="Needs Ultralytics and weights/yolo26m.pt, as on the GPU server.",
)
NOTEBOOK_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]


def test_a_detector_must_implement_detect():
    class Incomplete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_yolo_is_registered():
    assert "yolo" in DETECTORS.names()


def test_the_test_double_finds_a_moving_car():
    car = DETECTORS.create("test-moving-car", step=2).detect(Frame(3, 0.1, np.zeros((1, 1, 3), np.uint8)))[0]
    assert car.class_name == "car" and car.box.x1 == 16


@needs_yolo
def test_yolo_finds_the_same_boxes_as_the_notebooks(monkeypatch):
    from traffic_violation_detection.detection import DETECTOR_CONFIG, detect, load_detector

    monkeypatch.chdir(ROOT)
    notebook_model = load_detector()
    detector = DETECTORS.create("yolo", weights="yolo26m.pt", classes=NOTEBOOK_CLASSES)
    with VideoReader(ROOT / "data" / "video-test.mp4") as reader:
        for frame in reader.frames(0, 5):
            expected = [(d["class_id"], d["bbox_xyxy"], d["confidence"]) for d in detect(notebook_model, frame.image, DETECTOR_CONFIG)]
            assert [(d.class_id, d.box.as_list(), d.confidence) for d in detector.detect(frame)] == expected


@needs_yolo
def test_an_unknown_class_is_refused(monkeypatch):
    monkeypatch.chdir(ROOT)
    with pytest.raises(ValueError, match="no classes named \\['tractor'\\]"):
        DETECTORS.create("yolo", weights="yolo26m.pt", classes=["car", "tractor"])
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_detection.py -q
```

Expected: `3 failed, 2 passed, the first with: AssertionError: assert 'yolo' in ['test-moving-car']`

- [ ] **Step 3: Write the code**

Then replace `src/tvd/detection/__init__.py`, so that importing the package registers YOLO.

`src/tvd/detection/yolo.py`:

```python
"""Ultralytics YOLO models, behind ``BaseDetector``."""

from pathlib import Path

from tvd.detection.base import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame


@DETECTORS.register("yolo")
class YoloDetector(BaseDetector):
    """A YOLO model from Ultralytics.

    Boxes are rounded to 0.01 px and confidences to 4 decimals, as in the notebooks, so that results can be
    compared exactly with theirs.
    """

    def __init__(
        self,
        weights: str,
        image_size: int = 1280,
        confidence: float = 0.10,
        iou: float = 0.7,
        classes: list[str] | None = None,
        weights_dir: str = "weights",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError('The YOLO detector needs Ultralytics: pip install -e ".[detection]"') from error
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(Path(weights_dir) / weights))
        self._image_size = image_size
        self._confidence = confidence
        self._iou = iou
        self._class_ids = None if classes is None else self._ids_of(classes, weights)

    def _ids_of(self, classes: list[str], weights: str) -> list[int]:
        ids = {name: class_id for class_id, name in self._model.names.items()}
        unknown = [name for name in classes if name not in ids]
        if unknown:
            raise ValueError(f"{weights} has no classes named {unknown}")
        return sorted(ids[name] for name in classes)

    def detect(self, frame: Frame) -> list[Detection]:
        result = self._model.predict(
            frame.image, imgsz=self._image_size, conf=self._confidence, iou=self._iou, classes=self._class_ids, verbose=False
        )[0]
        boxes = result.boxes
        return [
            Detection(Box(*(round(float(v), 2) for v in xyxy)), int(class_id), self._model.names[int(class_id)], round(float(confidence), 4))
            for xyxy, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]
```

`src/tvd/detection/__init__.py`:

```python
"""Detectors. Importing this package registers every detector in it."""

from tvd.detection import yolo  # noqa: F401  Registers "yolo".
from tvd.detection.base import DETECTORS, BaseDetector

__all__ = ["DETECTORS", "BaseDetector"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `22 passed`.

If `test_yolo_finds_the_same_boxes_as_the_notebooks` fails, compare the first differing box: the usual cause is a different class list or image size from `DETECTOR_CONFIG` in `traffic_violation_detection/detection.py`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_detection.py src/tvd/detection/yolo.py src/tvd/detection/__init__.py
git commit -m "Add the YOLO detector"
```

### Task 6: Tracker interface

**Why:** This is the base class you described as `TrackingObjectBase`: `BaseTracker`. A tracker takes one frame's detections and returns the objects it tracks, and can forget everything with `reset()`. Nothing else may be needed from a tracker (**interface segregation**), so any tracker can replace any other (**Liskov substitution**). The test double `SameIdTracker` joins `conftest.py`.

**Files:**
- Replace: `tests/tvd/conftest.py`
- Create: `tests/tvd/test_tvd_tracking.py`
- Create: `src/tvd/tracking/base.py`
- Create: `src/tvd/tracking/__init__.py`

**Interfaces:**
- Consumes: `Registry` (Task 2), `Detection`, `Frame`, `TrackedObject` (Task 1).
- Produces: `BaseTracker.update(detections: list[Detection], frame: Frame) -> list[TrackedObject]` and `BaseTracker.reset() -> None` (both abstract), and `TRACKERS: Registry[BaseTracker]`. Test double `test-same-id`, which counts its resets.

- [ ] **Step 1: Write the failing tests**

`conftest.py` is replaced as a whole: it keeps what it had, and gains a test double.

`tests/tvd/conftest.py`:

```python
"""Test doubles for the tvd package. They register themselves, as real implementations do."""

from pathlib import Path

import pytest

from tvd.detection import DETECTORS, BaseDetector
from tvd.domain import Box, Detection, Frame, TrackedObject
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


@pytest.fixture
def sample_video() -> Path:
    """640x360, 24 FPS, 240 frames, committed in data/."""
    return ROOT / "data" / "traffic_sample.mp4"
```

`tests/tvd/test_tvd_tracking.py`:

```python
import pytest

from tvd.tracking import BaseTracker


def test_a_tracker_must_implement_update_and_reset():
    class UpdateOnly(BaseTracker):
        def update(self, detections, frame):
            return []

    with pytest.raises(TypeError):
        UpdateOnly()
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_tracking.py -q
```

Expected: `ImportError while loading conftest ... ModuleNotFoundError: No module named 'tvd.tracking'`

- [ ] **Step 3: Write the code**

`src/tvd/tracking/base.py`:

```python
"""What every tracker offers."""

from abc import ABC, abstractmethod

from tvd.domain import Detection, Frame, TrackedObject
from tvd.registry import Registry


class BaseTracker(ABC):
    """Links detections across frames, giving each object one track id.

    Any tracker can replace any other (the Liskov substitution principle): each takes a frame's detections
    and returns the objects it tracks in that frame, and nothing else in the pipeline knows which it is.
    """

    @abstractmethod
    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        """Advance by one frame, and return the objects tracked in it."""

    @abstractmethod
    def reset(self) -> None:
        """Forget every track, as at a cut in the video."""


TRACKERS: Registry[BaseTracker] = Registry("tracker")
```

`src/tvd/tracking/__init__.py`:

```python
"""Trackers. Importing this package registers every tracker in it."""

from tvd.tracking.base import TRACKERS, BaseTracker

__all__ = ["TRACKERS", "BaseTracker"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `23 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/conftest.py tests/tvd/test_tvd_tracking.py src/tvd/tracking/base.py src/tvd/tracking/__init__.py
git commit -m "Add the tracker interface"
```

### Task 7: BoxMOT trackers: ByteTrack, OC-SORT, BoT-SORT

**Why:** `BoxmotTracker` holds what every BoxMOT tracker shares: converting detections to BoxMOT's array and its output back. Each concrete tracker only says how to build itself (the **template method** pattern), in three lines. **This is how trackers are swapped:** `ByteTrackTracker`, `OcSortTracker` and `BotSortTracker` register under `bytetrack`, `ocsort` and `botsort`, and a settings file picks one. A new BoxMOT tracker later is one more small subclass, and no other code changes.

**Files:**
- Replace: `tests/tvd/test_tvd_tracking.py`
- Create: `src/tvd/tracking/boxmot_trackers.py`
- Replace: `src/tvd/tracking/__init__.py`

**Interfaces:**
- Consumes: `BaseTracker`, `TRACKERS` (Task 6).
- Produces: `BoxmotTracker(**options)` (abstract `_create(boxmot)`); `ByteTrackTracker`, `OcSortTracker`, `BotSortTracker`, registered as `"bytetrack"`, `"ocsort"`, `"botsort"`. Options go to `ByteTrackConfig`, `OcSortConfig`, `BotSortConfig` of BoxMOT 26.

- [ ] **Step 1: Write the failing test**

This replaces the test file of the previous task with its complete version.

`tests/tvd/test_tvd_tracking.py`:

```python
import importlib.util

import numpy as np
import pytest

from tvd.domain import Box, Detection, Frame
from tvd.tracking import TRACKERS, BaseTracker
from tvd.tracking.boxmot_trackers import OcSortTracker

needs_boxmot = pytest.mark.skipif(importlib.util.find_spec("boxmot") is None, reason="Needs BoxMOT, as on the GPU server.")
IMAGE = np.zeros((240, 320, 3), np.uint8)


def moving_car(index: int) -> list[Detection]:
    return [Detection(Box(10 + 5 * index, 50, 50 + 5 * index, 80), 2, "car", 0.9)]


def test_a_tracker_must_implement_update_and_reset():
    class UpdateOnly(BaseTracker):
        def update(self, detections, frame):
            return []

    with pytest.raises(TypeError):
        UpdateOnly()


def test_the_boxmot_trackers_are_registered():
    assert {"bytetrack", "ocsort", "botsort"} <= set(TRACKERS.names())


@needs_boxmot
@pytest.mark.parametrize("name, options", [("bytetrack", {}), ("ocsort", {"use_byte": True}), ("botsort", {"use_cmc": False, "use_embeddings": False})])
def test_a_moving_car_keeps_one_track_id(name, options):
    tracker = TRACKERS.create(name, **options)
    seen = [tracked for index in range(12) for tracked in tracker.update(moving_car(index), Frame(index, index / 30, IMAGE))]
    assert seen and {t.track_id for t in seen} == {seen[0].track_id}
    assert all(t.class_name == "car" and t.class_id == 2 for t in seen)


@needs_boxmot
def test_no_detections_give_no_tracked_objects():
    assert TRACKERS.create("ocsort").update([], Frame(0, 0.0, IMAGE)) == []


@needs_boxmot
def test_reset_builds_a_new_tracker():
    class CountingOcSort(OcSortTracker):
        built = 0

        def _create(self, boxmot):
            CountingOcSort.built += 1
            return super()._create(boxmot)

    tracker = CountingOcSort(use_byte=True)
    tracker.reset()
    assert CountingOcSort.built == 2
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_tracking.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.tracking.boxmot_trackers'`

- [ ] **Step 3: Write the code**

Then replace `src/tvd/tracking/__init__.py`, so that importing the package registers the three trackers.

`src/tvd/tracking/boxmot_trackers.py`:

```python
"""Trackers from BoxMOT, behind ``BaseTracker``."""

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from tvd.domain import Box, Detection, Frame, TrackedObject
from tvd.tracking.base import TRACKERS, BaseTracker


def import_boxmot():
    """BoxMOT, imported only when a tracker is built, with its log messages limited to warnings."""
    try:
        import boxmot
        import boxmot.utils  # noqa: F401  Imported so that its logger exists before it is quietened.
    except ImportError as error:
        raise RuntimeError('BoxMOT trackers need BoxMOT: pip install -e ".[tracking]" && pip install --no-deps boxmot==26.0.0') from error
    logging.getLogger("boxmot").setLevel(logging.WARNING)
    return boxmot


class BoxmotTracker(BaseTracker):
    """What every BoxMOT tracker shares: converting detections to BoxMOT's array and its output back.

    A subclass only says how to build its own tracker (the template method pattern), from the options
    of its settings.
    """

    def __init__(self, **options: Any) -> None:
        self._options = options
        self._tracker = self._create(import_boxmot())

    @abstractmethod
    def _create(self, boxmot: Any) -> Any:
        """A new BoxMOT tracker, built from ``self._options``."""

    def reset(self) -> None:
        self._tracker = self._create(import_boxmot())

    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        rows = np.array([[*d.box.as_list(), d.confidence, d.class_id] for d in detections], dtype=np.float32).reshape(-1, 6)
        output = self._tracker.update(rows, frame.image)
        tracked = []
        for row in output if len(output) else []:
            matched = detections[int(row[7])]  # BoxMOT's last column is the index of the detection matched in this frame.
            box = Box(*(round(float(v), 2) for v in row[:4]))
            tracked.append(TrackedObject(int(row[4]), box, matched.class_id, matched.class_name, round(float(row[5]), 4)))
        return tracked


@TRACKERS.register("bytetrack")
class ByteTrackTracker(BoxmotTracker):
    def _create(self, boxmot: Any) -> Any:
        return boxmot.ByteTrack(boxmot.ByteTrackConfig(**self._options))


@TRACKERS.register("ocsort")
class OcSortTracker(BoxmotTracker):
    def _create(self, boxmot: Any) -> Any:
        return boxmot.OcSort(boxmot.OcSortConfig(**self._options))


@TRACKERS.register("botsort")
class BotSortTracker(BoxmotTracker):
    def _create(self, boxmot: Any) -> Any:
        return boxmot.BotSort(boxmot.BotSortConfig(**self._options))
```

`src/tvd/tracking/__init__.py`:

```python
"""Trackers. Importing this package registers every tracker in it."""

from tvd.tracking import boxmot_trackers  # noqa: F401  Registers "bytetrack", "ocsort" and "botsort".
from tvd.tracking.base import TRACKERS, BaseTracker

__all__ = ["TRACKERS", "BaseTracker"]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `29 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_tracking.py src/tvd/tracking/boxmot_trackers.py src/tvd/tracking/__init__.py
git commit -m "Add the ByteTrack, OC-SORT and BoT-SORT trackers"
```

### Task 8: Track history

**Why:** A tracker answers one frame at a time, but the rules need each vehicle's whole path. `TrackHistory` collects it (**single responsibility**). The class of a track is the one reported most often, because a detector sometimes calls a van a truck for a frame or two.

**Files:**
- Create: `tests/tvd/test_tvd_history.py`
- Create: `src/tvd/tracking/history.py`

**Interfaces:**
- Consumes: `Box`, `TrackedObject` (Task 1).
- Produces: `TrackPoint(frame_index, box, confidence)`; `Track` with `track_id`, `points`, `class_name`, `first_frame`, `last_frame`, `box_at(frame_index) -> Box | None`; `TrackHistory` with `add(frame_index, objects)`, `tracks -> dict[int, Track]`, `frames_seen`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_history.py`:

```python
from tvd.domain import Box, TrackedObject
from tvd.tracking.history import TrackHistory


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
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_history.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.tracking.history'`

- [ ] **Step 3: Write the code**

`src/tvd/tracking/history.py`:

```python
"""Tracks as they build up over a video."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from tvd.domain import Box, TrackedObject


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    box: Box
    confidence: float


@dataclass
class Track:
    """Every frame in which one track id was reported."""

    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def class_name(self) -> str:
        """The class reported most often; on a tie, the one reported first."""
        return self.class_counts.most_common(1)[0][0]

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
            track.points.append(TrackPoint(frame_index, tracked.box, tracked.confidence))
            track.class_counts[tracked.class_name] += 1

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `34 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_history.py src/tvd/tracking/history.py
git commit -m "Add the track history"
```

### Task 9: Detect and track a video

**Why:** `track_video` joins the parts, and depends only on the abstractions (**dependency inversion**): any detector and any tracker work. The last test runs your new code and the notebooks' code on the first 150 frames of `video-test.mp4`, and requires the same track ids and boxes in every frame.

**Files:**
- Create: `tests/tvd/test_tvd_perception.py`
- Create: `src/tvd/perception.py`

**Interfaces:**
- Consumes: `VideoReader` (Task 3), `BaseDetector` (Task 4), `BaseTracker` (Task 6), `TrackHistory` (Task 8); `YoloDetector` and `OcSortTracker` in the test.
- Produces: `track_video(reader, detector, tracker, track_classes: set[str], start=0, stop=None) -> TrackHistory`. It resets the tracker first.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_perception.py`:

```python
import importlib.util

import pytest
from conftest import ROOT

from tvd.detection import DETECTORS
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

Expected: `E   ModuleNotFoundError: No module named 'tvd.perception'`

- [ ] **Step 3: Write the code**

`src/tvd/perception.py`:

```python
"""Detection and tracking over a video."""

from tvd.detection import BaseDetector
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
) -> TrackHistory:
    """Detect every frame from ``start`` to ``stop``, and track the detections of ``track_classes``.

    It depends only on the two abstractions, so any detector and any tracker work here.
    """
    tracker.reset()
    history = TrackHistory()
    for frame in reader.frames(start, stop):
        detections = [d for d in detector.detect(frame) if d.class_name in track_classes]
        history.add(frame.index, tracker.update(detections, frame))
    return history
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `37 passed`.

`test_the_same_tracks_as_the_notebooks` is the proof that Tasks 1 to 9 rebuilt detection and tracking without changing a result. It takes about 10 s on the server.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_perception.py src/tvd/perception.py
git commit -m "Detect and track a video through the interfaces"
```

### Task 10: Settings files

**Why:** Nothing a user might tune is written in code. Settings are YAML, so each one can carry a comment, and pydantic checks them. `extra="forbid"` refuses a misspelt key instead of ignoring it, and every problem is reported at once with its place in the file. `configs/tvd/perception.yaml` holds the settings of Notebook 03.

**Files:**
- Create: `tests/tvd/test_tvd_config.py`
- Create: `src/tvd/config.py`
- Create: `configs/tvd/perception.yaml`

**Interfaces:**
- Consumes: Nothing from earlier tasks. The test names the test doubles only as text.
- Produces: `ConfigError(ValueError)`; `ComponentConfig(type: str, options: dict = {})`, where `options:` left empty means no options; `PerceptionConfig(detector, tracker, track_classes: list[str])`, frozen; `load_config(path, model) -> model`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_config.py`:

```python
import pytest
from conftest import ROOT
from pydantic import ValidationError

from tvd.config import ConfigError, PerceptionConfig, load_config

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
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_config.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.config'`

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
        return model.model_validate(data if data is not None else {})
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(str(part) for part in e['loc']) or '(file)'}: {e['msg']}" for e in error.errors())
        raise ConfigError(f"{path}: {problems}") from error
```

`configs/tvd/perception.yaml`:

```yaml
# How the service detects and tracks. Change "type" to swap an implementation; no code changes.
detector:
  type: yolo
  options:
    weights: yolo26m.pt
    image_size: 1280
    confidence: 0.10        # low on purpose: the tracker uses low-score boxes
    iou: 0.7
    classes: [person, bicycle, car, motorcycle, bus, truck, traffic light]
tracker:
  type: ocsort              # one of: bytetrack, ocsort, botsort
  options:
    use_byte: true          # OC-SORT with BYTE's low-score association, as in Notebook 03
track_classes: [bicycle, car, motorcycle, bus, truck]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `48 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_config.py src/tvd/config.py configs/tvd/perception.yaml
git commit -m "Add settings files checked by pydantic"
```

### Task 11: Factory

**Why:** The factory is the only place where a name in a file becomes an object. Everything else receives ready-made components (**dependency inversion**). A wrong name or option becomes a `ConfigError` that says which component it was.

**Files:**
- Create: `tests/tvd/test_tvd_factory.py`
- Create: `src/tvd/factory.py`

**Interfaces:**
- Consumes: `DETECTORS` (Task 4/5), `TRACKERS` (Task 6/7), `ComponentConfig`, `ConfigError` (Task 10).
- Produces: `build(registry, config) -> T`, `build_detector(config: ComponentConfig) -> BaseDetector`, `build_tracker(config: ComponentConfig) -> BaseTracker`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_factory.py`:

```python
import importlib.util

import pytest

from tvd.config import ComponentConfig, ConfigError
from tvd.factory import build_detector, build_tracker

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
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_factory.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.factory'`

- [ ] **Step 3: Write the code**

`src/tvd/factory.py`:

```python
"""Builds components from their settings: the one place where names in a file become objects.

The pipeline receives a detector and a tracker and never builds them itself (the dependency inversion
principle), so changing either is a change of settings, not of code.
"""

from typing import TypeVar

from tvd.config import ComponentConfig, ConfigError
from tvd.detection import DETECTORS, BaseDetector
from tvd.registry import Registry
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
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `52 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_factory.py src/tvd/factory.py
git commit -m "Build components from their settings"
```

### Task 12: Command line

**Why:** The command line is where everything is assembled (the **composition root**): it reads the settings, builds the components and runs `track_video`. It opens the video before loading any model, so a wrong path fails in a fraction of a second. Errors print as one line, with exit code 2.

**Files:**
- Create: `tests/tvd/test_tvd_cli.py`
- Create: `src/tvd/cli.py`

**Interfaces:**
- Consumes: `load_config`, `PerceptionConfig` (Task 10), `build_detector`, `build_tracker` (Task 11), `track_video` (Task 9), `VideoReader` (Task 3).
- Produces: `python -m tvd.cli track --video <file> --config <yaml> --output <json>`; `run_track(video, config_path, output) -> dict`; `tracks_as_json(history) -> list[dict]`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

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
```

- [ ] **Step 2: Run the test and see it fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_cli.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.cli'`

- [ ] **Step 3: Write the code**

`src/tvd/cli.py`:

```python
"""Command line: ``python -m tvd.cli track --video <file> --config <yaml> --output <json>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tvd.config import ConfigError, PerceptionConfig, load_config
from tvd.factory import build_detector, build_tracker
from tvd.perception import track_video
from tvd.tracking.history import TrackHistory
from tvd.video import VideoReader


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tvd")
    commands = parser.add_subparsers(dest="command", required=True)
    track = commands.add_parser("track", help="Detect and track the vehicles of a video")
    track.add_argument("--video", type=Path, required=True)
    track.add_argument("--config", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_track(args.video, args.config, args.output)
    except (ConfigError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{summary['video']}: {summary['frames']} frames, {summary['tracks']} tracks, detector {summary['detector']}, tracker {summary['tracker']}, {summary['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `54 passed`.

- [ ] **Step 5: Track a whole video with OC-SORT**

```bash
scripts/server.sh run 'python -m tvd.cli track --video data/video-test.mp4 --config configs/tvd/perception.yaml --output outputs/tvd/video-test_tracks.json'
```

Expected, after some log lines from BoxMOT (the time varies):

```text
video-test.mp4: 900 frames, 136 tracks, detector yolo, tracker ocsort, 22.7 s
```

136 tracks is what Notebook 03 found with OC-SORT and BYTE.

- [ ] **Step 6: Swap the tracker without touching any code**

In `configs/tvd/perception.yaml`, change the tracker to ByteTrack, and leave `options:` empty:

```yaml
tracker:
  type: bytetrack
  options:
```

```bash
scripts/server.sh sync && scripts/server.sh run 'python -m tvd.cli track --video data/video-test.mp4 --config configs/tvd/perception.yaml --output outputs/tvd/video-test_tracks.json'
```

Expected: `video-test.mp4: 900 frames, 154 tracks, detector yolo, tracker bytetrack, 21.0 s`

Then BoT-SORT, without its appearance model:

```yaml
tracker:
  type: botsort
  options:
    use_embeddings: false
```

Expected: `video-test.mp4: 900 frames, 152 tracks, detector yolo, tracker botsort, 22.1 s`

Put OC-SORT back (`type: ocsort`, `use_byte: true`) before the next step. Three trackers, one word changed each time, and no code changed: that is the open/closed principle at work.

- [ ] **Step 7: See how mistakes are reported**

A tracker that does not exist (`type: deepsort`):

```text
error: deepsort: Unknown tracker 'deepsort'. Available: botsort, bytetrack, ocsort
```

A misspelt option (`use_bytes: true`):

```text
error: ocsort: OcSortConfig.__init__() got an unexpected keyword argument 'use_bytes'
```

A video that does not exist (`--video data/missing.mp4`), reported in about 0.3 s, before any model loads:

```text
error: Video not found: data/missing.mp4
```

Each exits with code 2: check with `echo $?` inside the same `run` command.

- [ ] **Step 8: Commit**

```bash
git add tests/tvd/test_tvd_cli.py src/tvd/cli.py
git commit -m "Add the track command"
```

---

## When Part 1 is done

- [ ] `scripts/server.sh test tests/tvd -q` gives `54 passed`, and `scripts/server.sh test -q` still passes the old tests too.
- [ ] `track` gives 136 tracks on `video-test.mp4` with OC-SORT, and changing one word in the YAML switches to ByteTrack or BoT-SORT.
- [ ] You can explain where each SOLID principle appears in `src/tvd/`, using the table above.

**Exercise, if you want one:** BoxMOT also has `DeepOcSort` and `StrongSort`. Each needs an appearance (ReID) model, which is in `weights/osnet_x0_25_msmt17.pt` on the server. Adding one is a new subclass of `BoxmotTracker`, registered under a new name. Read BoxMOT's `DeepOcSortConfig` to find its options. This exercise was not checked while writing the plan.

**Next:** the plan for Part 2, Scene and rules, is written once Part 1 is done.
