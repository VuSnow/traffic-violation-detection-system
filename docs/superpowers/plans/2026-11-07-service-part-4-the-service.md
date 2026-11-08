# Service Part 4: The Service — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Parts 1 to 3 into a service that a backend can call: upload a video with its camera and the time of its first frame, follow the job, and get its records and evidence over REST, within 2 hours.

**Architecture:** A core Python API, `ReviewService`, checks each upload and queues it in SQLite, or refuses it with a reason; a worker thread runs the queued jobs one at a time with models loaded once (`Pipeline`), using Part 2's `check_video` and Part 3's `review_check`; a thin FastAPI app translates between HTTP and the core. Every setting is in `configs/service.yaml`, including the measured speed that sets the longest video and refuses an upload that would miss the deadline.

**Tech Stack:** Python 3.11, FastAPI 0.162 (MIT) with Starlette and python-multipart, uvicorn 0.54 (BSD), SQLite (the standard library), httpx for tests; with Parts 1 to 3 (PyTorch, ONNX Runtime, OpenCV 5.0.0, NumPy 2.4.6, pydantic 2).

**Spec:** `docs/superpowers/specs/2026-11-07-review-service-design.md` (Sections 2, 4, 5, 6, 7, 8, 9 and 10).

**Verified:** every file below, the state before each task's code (with the failure it shows) and the state after it were run on ai_server_2 before this plan was written. After the last task: 340 tests pass in `tests/tvd`, and the real service on the server took an upload of `traffic_video_modified.mp4` through to its two records, with the same clips as Notebook 08.

## The five parts

| Part | Builds | Plan |
|---|---|---|
| 1. Foundations and perception | Data types, registry, video reader, detector, trackers, settings, `track` command | Done: `2026-11-07-service-part-1-foundations-and-perception.md` |
| 2. Scene and rules | Camera files, rule settings, camera motion, signal reading, the wrong-way and red-light rules, `check` command | Done: `2026-11-07-service-part-2-scene-and-rules.md` |
| 3. Plates and evidence | Clips, plate models in a process of their own, voting, evidence images, records of schema 2, `review` command | Done: `2026-11-07-service-part-3-plates-and-evidence.md` |
| **4. The service** | Service settings, camera catalog, upload checks, job queue in SQLite, job folders, a worker, a REST API, `serve` command | This document |
| 5. Deployment and MLOps | Docker, running on ai_server_2, logging, health checks, model versions | Later |

## What one upload does

```text
backend ── POST /jobs (video, camera_id, recorded_at) ──> api ──> ReviewService.submit
                                                                   │ recorded_at has a time zone; the camera exists
                                                                   │ save_upload ──> <storage>/<job id>/video.mp4  (413 if too large)
                                                                   │ inspect_video: a video of this camera's size; its reference frame exists
                                                                   │ estimate_seconds: measured speed × length ≤ deadline, behind the waiting jobs (503 if not)
                                                                   └──> JobStore (SQLite): queued        ── 202 and the job id
Worker thread, one job at a time ──> ReviewService.run_next ──> Pipeline.run (models loaded once)
                                       check_video (Part 2) + FrameCounter ──> progress: tracking, camera motion, evidence
                                       review_check (Part 3) ──> <storage>/<job id>/records.json, clips/, evidence/, plates/
                                       done, or failed with its reason
backend ── GET /jobs/{id}, /jobs/{id}/records, /jobs/{id}/files/{path}, /cameras ──> api ──> ReviewService
```

## Global Constraints

- Python 3.11, in the `tvd` environment: on ai_server_2, `~/workspace/dungvu/envs/tvd`. The new packages are FastAPI, uvicorn and python-multipart (and httpx, for tests), added to `pyproject.toml` in Task 0.
- Never let pip change OpenCV (5.0.0.93), NumPy (2.4.6) or pydantic (2.14.3).
- New code goes in `src/tvd/service/`, and its tests in `tests/tvd/test_tvd_*.py`. The notebooks' package `traffic_violation_detection` does not change.
- **Nothing that a user might tune is written in code.** The service's settings are in `configs/service.yaml`, with no defaults in code; each job uses the settings files of Parts 1 to 3, which it names.
- **Settings are read and checked when the service starts** (spec, Section 4), camera files included: a wrong one stops the service from starting, naming the file. A file changed while the service runs takes effect when it starts again, so every job of one run uses the same settings, and each record keeps their versions.
- **Results within 2 hours of upload** (spec, Section 2): an upload that would miss the deadline, alone or behind the waiting jobs, is refused, never queued to be late.
- **An upload that is refused leaves nothing behind**: no job, no folder, and a reason the backend can show.
- **The service never confirms a violation**: every record is `pending`, as in Part 3.
- **Jobs run one at a time**, in order of arrival: they share the GPU.
- **ai_server_2 is shared.** The service listens on `127.0.0.1` by default, and the tests never open a port. Port 8000 is taken; the service uses 8040.
- **Plate numbers are private**: records and plate images stay in the job folders, under `outputs/`, which is git-ignored. Nothing in this plan prints a plate.
- Tests that need the GPU stack, the plate models or the videos in `data/` skip themselves where those are missing. The counts below are from the GPU server.

## Review Focus

- **Two uploads at the same moment** must not both take the last room before the deadline: the check and the queuing happen under one lock, `test_an_upload_that_would_miss_the_deadline_is_refused` (Task 10).
- **A restart in the middle of a job** must not leave it `running` for ever, nor lose the waiting ones: `test_after_a_restart_the_job_that_was_running_has_failed_and_the_queue_is_kept` (Task 10), `test_after_a_restart_a_job_that_was_running_has_failed` (Task 4).
- **A path that leads outside a job's folder**, such as `..%2Fjobs.sqlite`, or a link inside it, must serve nothing: `test_only_files_inside_the_job_folder_are_given` (Task 5), `test_an_unknown_job_or_file_is_not_found` (Tasks 10 and 12).
- **A second service started on the same storage** would run every job twice: it is refused, `test_a_second_service_on_the_same_storage_is_refused` (Task 10).
- **A failing job, or a hung plate model,** must not stop the queue: `test_a_job_that_fails_keeps_its_reason_and_the_queue_goes_on` (Task 10), `test_the_worker_outlives_an_unexpected_error` (Task 11), `test_models_that_do_not_answer_in_time_are_stopped` (Task 8).

## How you will work

As in Parts 1 to 3: edit on your laptop, run on ai_server_2 with `scripts/server.sh`. Tests of the service write their storage in pytest's temporary folders, and use the test detector, tracker and plate models with the committed sample video, so most of them need no GPU. Only Task 13's last steps start the real service on the server.

## File structure

```text
src/tvd/service/
├── __init__.py               Task 1    the package
├── settings.py               Task 1    ServiceSettings, load_service_settings
├── cameras.py                Task 2    Camera, CameraCatalog
├── jobs.py                   Task 4    Job, JobStore, the four states
├── storage.py                Task 5    JobFolders
├── intake.py                 Task 6    Refused, save_upload, VideoFacts, inspect_video, estimate_seconds
├── runner.py                 Task 9    Pipeline
├── core.py                   Task 10   ReviewService, NotFound, NotReady
├── worker.py                 Task 11   Worker
└── api.py                    Task 12   create_app
src/tvd/motion/base.py, orb.py Task 3   check_frames                                       (changed)
src/tvd/review.py             Task 5    prepare_folder accepts the upload beside the review   (changed)
src/tvd/observers.py          Task 7    FrameCounter                                        (changed)
src/tvd/check.py              Task 7    check_video(..., observers)                         (changed)
src/tvd/plates/base.py, onnx.py Task 8  is_alive, timeout_seconds                           (changed)
src/tvd/cli.py                Task 3/5/13 check_frames before work; serve                   (changed)
configs/service.yaml          Task 1    storage, jobs, server, files
configs/tvd/plates.yaml       Task 8    timeout_seconds                                     (changed)
pyproject.toml                Task 0    fastapi, uvicorn, python-multipart, httpx           (changed)
tests/tvd/service_data.py     Task 9/10 the settings files of a test service
```

**How the parts depend on each other.** Arrows point at what a module uses. `cli.py` builds the service; `api.py` only knows the core; the core only knows the parts below it.

```text
cli (serve) ──> api ──> core ──> intake ──> cameras, settings, Part 1's VideoReader, Part 2's check_frame_size
                 │        ├──> jobs (SQLite), storage (folders)
                 │        └──> runner ──> Part 2's check_video, Part 3's review_check, observers.FrameCounter
                 └──> worker ──> core
```

## SOLID and patterns in this part

| Principle or pattern | Where |
|---|---|
| **S**ingle responsibility | `jobs.py` keeps states, `storage.py` folders, `intake.py` checks uploads, `runner.py` runs one job, `core.py` puts them together, `worker.py` loops, `api.py` speaks HTTP |
| **O**pen/closed | Progress is one more `FrameObserver`; `check_video` only learned to take observers. A new kind of camera motion needs one measured speed in `service.yaml`, and no code |
| **L**iskov substitution | `check_frames` is in every motion estimator, with a default that accepts any video |
| **I**nterface segregation | The API sees only `ReviewService`'s calls, never the store or the pipeline |
| **D**ependency inversion | `ReviewService` takes its settings, a clock and a pipeline; the tests give it a clock of their own |
| Facade | `ReviewService` is the one door to jobs, folders, checks and the pipeline |
| Repository | `JobStore` hides SQLite behind methods named after what a job does |
| Producer and consumer | Uploads add jobs to the queue; one worker takes them |

---

### Task 0: Branch, and the REST layer's packages

**Why:** The spec chose FastAPI and uvicorn for the REST layer (Section 10). FastAPI needs python-multipart to read an uploaded file, and its test client sends requests with httpx. They go in the `tvd` environment of the server, and must not change what Parts 1 to 3 rely on.

**Files:**
- Change: `pyproject.toml`

- [ ] **Step 1: Start the branch from `main`, where Part 3 is merged**

```bash
git checkout main
git checkout -b service/part-4
```

- [ ] **Step 2: Name the new packages**

In `pyproject.toml`:

Replace:

```toml
# For the review service (src/tvd). pydantic (MIT) checks settings files; PyYAML (MIT) reads them.
service = [
    "pydantic>=2.6,<3",
    "pyyaml>=6.0,<7",
]
dev = [
    "pytest>=8",
]
```

with:

```toml
# For the review service (src/tvd). pydantic (MIT) checks settings files; PyYAML (MIT) reads them.
# FastAPI (MIT) and uvicorn (BSD) serve the REST API; python-multipart (Apache-2.0) reads uploaded files.
service = [
    "pydantic>=2.6,<3",
    "pyyaml>=6.0,<7",
    "fastapi>=0.115,<1",
    "uvicorn>=0.32,<1",
    "python-multipart>=0.0.20",
]
# httpx (BSD) is what FastAPI's test client sends requests with.
dev = [
    "pytest>=8",
    "httpx>=0.28,<1",
]
```

- [ ] **Step 3: Install them on the server, and check that nothing else changed**

pip prints what it would change first: it must not list OpenCV, NumPy or pydantic.

```bash
scripts/server.sh run 'pip install --dry-run "fastapi>=0.115,<1" "uvicorn>=0.32,<1" "python-multipart>=0.0.20" "httpx>=0.28,<1" | tail -1'
```

Expected: `Would install annotated-doc-0.0.5 fastapi-0.162.0 opentelemetry-api-1.45.2 python-multipart-0.0.33 starlette-1.10.0 uvicorn-0.54.0` (httpx is already there; the versions may be newer). Then install them, and check:

```bash
scripts/server.sh run 'pip install -q "fastapi>=0.115,<1" "uvicorn>=0.32,<1" "python-multipart>=0.0.20" "httpx>=0.28,<1"'
scripts/server.sh run 'python -c "import cv2, numpy, pydantic, fastapi, uvicorn, multipart; print(cv2.__version__, numpy.__version__, pydantic.VERSION, fastapi.__version__)"'
```

Expected: `5.0.0 2.4.6 2.14.3 0.162.0`: OpenCV, NumPy and pydantic as before.

- [ ] **Step 4: Check that Part 3 still passes**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `263 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "Add the REST layer's packages"
```

### Task 1: The service's settings

**Why:** Everything the service needs to know about itself is in `configs/service.yaml` (spec, Section 4): where jobs are kept and for how long, the deadline, where it listens, and which settings files the pipeline reads. **The longest video comes from a measurement, not a guess** (Section 7): the work per second of video of each kind of camera, measured on ai_server_2 for this plan and multiplied by 1.5. The longest video is then what fits in the deadline: about 55 minutes for a fixed camera and 17 for a moving one.

**Files:**
- Create: `tests/tvd/test_tvd_service_settings.py`
- Create: `configs/service.yaml`
- Create: `src/tvd/service/settings.py`
- Create: `src/tvd/service/__init__.py`

**Interfaces:**
- Consumes: `StrictModel`, `load_config` (Part 1).
- Produces: `ServiceSettings(version, storage, jobs, server, files)`; `StorageSettings(folder, keep_days, max_upload_mb)`; `JobSettings(deadline_minutes, seconds_per_video_second)` with `work_seconds(motion_type, video_seconds)` and `longest_video_seconds(motion_type)`; `ServerSettings(host, port)`; `SettingsFiles(cameras, perception, rules, plates, evidence)`; `load_service_settings(path)`.

- [ ] **Step 1: Write the failing test**

**The service listens on `127.0.0.1` by default.** It has no authentication (a non-goal of the spec), and ai_server_2 is shared, so it is reachable from the server only. Change `host` to `0.0.0.0` when your backend runs on another machine, or reach it through `ssh -L 8040:localhost:8040 20.0.14.182`.

`tests/tvd/test_tvd_service_settings.py`:

```python
import pytest
from conftest import ROOT

from tvd.config import ConfigError
from tvd.service.settings import load_service_settings

SERVICE = ROOT / "configs" / "service.yaml"


def test_the_service_file_is_read():
    settings = load_service_settings(SERVICE)
    assert (settings.server.host, settings.server.port, settings.jobs.deadline_minutes) == ("127.0.0.1", 8040, 120)
    assert settings.files.cameras.as_posix() == "configs/cameras" and settings.storage.keep_days == 7


def test_the_longest_video_follows_the_measured_speed():
    jobs = load_service_settings(SERVICE).jobs
    assert round(jobs.longest_video_seconds("fixed") / 60) == 55 and round(jobs.longest_video_seconds("orb") / 60) == 17
    assert jobs.work_seconds("orb", 60) == pytest.approx(420)


@pytest.mark.parametrize("old, new, message", [
    ("port: 8040", "port: 80400", "server.port: Input should be less than or equal to 65535"),
    ("    orb: 7.0", "    orb: 0", "jobs.seconds_per_video_second: Value error, every value must be greater than 0"),
    ("  keep_days: 7", "  keep_day: 7", "storage.keep_days: Field required"),
])
def test_a_wrong_setting_is_named(tmp_path, old, new, message):
    path = tmp_path / "service.yaml"
    path.write_text(SERVICE.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=message):
        load_service_settings(path)
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_service_settings.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service'`

- [ ] **Step 3: Write the code**

`configs/service.yaml`:

```yaml
# The review service: where it keeps jobs, how long a video may be, and where it listens.
# Paths are relative to the folder the service is started from: the project folder.
version: 1
storage:
  folder: outputs/service       # each job gets <folder>/<job id>/; outputs/ is not in git
  keep_days: 7                  # a finished job's folder and record are deleted after this
  max_upload_mb: 4096           # a larger upload is refused while it arrives
jobs:
  deadline_minutes: 120         # results within 2 hours of upload, including the wait in the queue
  seconds_per_video_second:     # work per second of video, by the camera's motion type: measured on ai_server_2 for
    fixed: 2.2                  #   Part 4 (1080p, 38.7 s for 27.2 s of video), times 1.5 for margin: about 55 min at most
    orb: 7.0                    #   (camera motion, 127.3 s for 27.2 s), times 1.5: about 17 min at most
server:
  host: 127.0.0.1               # only this machine; 0.0.0.0 opens it to the network, with no authentication
  port: 8040                    # 8000 is taken on ai_server_2
files:
  cameras: configs/cameras      # one <camera id>.yaml per camera
  perception: configs/tvd/perception.yaml
  rules: configs/tvd/rules.yaml
  plates: configs/tvd/plates.yaml
  evidence: configs/tvd/evidence.yaml
```

`src/tvd/service/settings.py`:

```python
"""The service's own settings: ``configs/service.yaml``. There are no defaults in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from tvd.config import StrictModel, load_config


class StorageSettings(StrictModel):
    folder: Path
    keep_days: float = Field(gt=0)
    max_upload_mb: int = Field(gt=0)


class JobSettings(StrictModel):
    deadline_minutes: float = Field(gt=0)
    seconds_per_video_second: dict[str, float] = Field(min_length=1)

    @field_validator("seconds_per_video_second")
    @classmethod
    def positive(cls, value: dict[str, float]) -> dict[str, float]:
        if any(v <= 0 for v in value.values()):
            raise ValueError("every value must be greater than 0")
        return value

    def work_seconds(self, motion_type: str, video_seconds: float) -> float:
        """How long a video takes, from the measured speed of its kind of camera."""
        return video_seconds * self.seconds_per_video_second[motion_type]

    def longest_video_seconds(self, motion_type: str) -> float:
        """The longest video of a kind of camera that can be done within the deadline, with an empty queue."""
        return self.deadline_minutes * 60 / self.seconds_per_video_second[motion_type]


class ServerSettings(StrictModel):
    host: str
    port: int = Field(ge=1, le=65535)


class SettingsFiles(StrictModel):
    cameras: Path
    perception: Path
    rules: Path
    plates: Path
    evidence: Path


class ServiceSettings(StrictModel):
    version: int
    storage: StorageSettings
    jobs: JobSettings
    server: ServerSettings
    files: SettingsFiles


def load_service_settings(path: str | Path) -> ServiceSettings:
    return load_config(path, ServiceSettings)
```

`src/tvd/service/__init__.py`:

```python
"""The review service: jobs from uploaded videos, processed one at a time, behind a REST API."""
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `268 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_service_settings.py configs/service.yaml src/tvd/service/settings.py src/tvd/service/__init__.py
git commit -m "Read the service's settings"
```

### Task 2: Every camera, checked when the service starts

**Why:** A camera file with a mistake would only fail when a video of that camera arrives, perhaps at night. So the service reads every file of `configs/cameras/` when it starts, builds its rules and its motion estimator, and refuses to start if any is wrong, naming the file. A file must be named after its camera, so that a `camera_id` always leads to one file.

**Files:**
- Create: `tests/tvd/test_tvd_service_cameras.py`
- Create: `src/tvd/service/cameras.py`

**Interfaces:**
- Consumes: `load_camera`, `CameraConfig` (Part 2); `build_rules`, `build_motion_estimator` (Part 2); `file_version` (Part 3).
- Produces: `Camera(config, path, version)` with `camera_id`, `motion_type` and `describe()`; `CameraCatalog(folder, rule_defaults, speeds)` with `get(camera_id) -> Camera | None`, `ids` and `describe()`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_service_cameras.py`:

```python
import shutil

import pytest
from conftest import ROOT

from tvd.camera import load_rule_defaults
from tvd.config import ConfigError
from tvd.service.cameras import CameraCatalog

CAMERAS = ROOT / "configs" / "cameras"
DEFAULTS = load_rule_defaults(ROOT / "configs" / "tvd" / "rules.yaml")
SPEEDS = {"fixed": 2.2, "orb": 7.0}


def copy_cameras(tmp_path):
    folder = tmp_path / "cameras"
    shutil.copytree(CAMERAS, folder)
    return folder


def test_every_camera_file_is_loaded():
    catalog = CameraCatalog(CAMERAS, DEFAULTS, SPEEDS)
    assert catalog.ids == ["rlr_occasions-C", "traffic_video_modified", "video-test"]
    camera = catalog.get("traffic_video_modified")
    assert camera.motion_type == "orb" and camera.version.startswith("sha256:") and catalog.get("nowhere") is None


def test_each_camera_is_described_for_the_backend():
    assert CameraCatalog(CAMERAS, DEFAULTS, SPEEDS).describe()[0] == {
        "camera_id": "rlr_occasions-C", "frame_size": [1280, 720], "motion": "fixed", "moving": False, "rules": ["red_light"],
    }


def test_a_file_named_after_another_camera_is_refused(tmp_path):
    folder = copy_cameras(tmp_path)
    (folder / "video-test.yaml").rename(folder / "junction.yaml")
    with pytest.raises(ConfigError, match="camera_id 'video-test' must be the file's name, 'junction'"):
        CameraCatalog(folder, DEFAULTS, SPEEDS)


def test_a_wrong_rule_setting_names_the_file_and_the_rule(tmp_path):
    folder = copy_cameras(tmp_path)
    path = folder / "traffic_video_modified.yaml"
    path.write_text(path.read_text().replace("pass_margin: 0.2", "pass_margn: 0.2"))
    with pytest.raises(ConfigError, match=r"traffic_video_modified.yaml: rules.red_light: .*pass_margn"):
        CameraCatalog(folder, DEFAULTS, SPEEDS)


def test_a_camera_whose_speed_was_never_measured_is_refused():
    with pytest.raises(ConfigError, match="no measured speed for a 'orb' camera"):
        CameraCatalog(CAMERAS, DEFAULTS, {"fixed": 2.2})


def test_a_folder_without_cameras_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="No camera file"):
        CameraCatalog(tmp_path, DEFAULTS, SPEEDS)
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_service_cameras.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.cameras'`

- [ ] **Step 3: Write the code**

`src/tvd/service/cameras.py`:

```python
"""Every camera the service knows: the files of one folder, all checked when the service starts.

A camera file with a mistake would only fail when a video of that camera arrives, perhaps at night.
So the service refuses to start instead, and names the file and the setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tvd.camera import CameraConfig, load_camera
from tvd.config import ConfigError, file_version
from tvd.factory import build_motion_estimator, build_rules


@dataclass(frozen=True)
class Camera:
    config: CameraConfig
    path: Path
    version: str

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def motion_type(self) -> str:
        return self.config.motion.type

    def describe(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id, "frame_size": list(self.config.frame_size), "motion": self.motion_type,
            "moving": self.motion_type != "fixed", "rules": sorted(self.config.rules),
        }


class CameraCatalog:
    """The cameras of ``folder``, one ``<camera id>.yaml`` each. ``speeds`` are the measured speeds by motion type."""

    def __init__(self, folder: Path, rule_defaults: dict[str, dict[str, Any]], speeds: dict[str, float]) -> None:
        if not folder.is_dir():
            raise ConfigError(f"Camera folder not found: {folder}")
        cameras = {}
        for path in sorted(folder.glob("*.yaml")):
            config = load_camera(path)
            if config.camera_id != path.stem:
                raise ConfigError(f"{path}: camera_id {config.camera_id!r} must be the file's name, {path.stem!r}")
            try:
                build_rules(config, rule_defaults)
                build_motion_estimator(config.motion)
            except ConfigError as error:
                raise ConfigError(f"{path}: {error}") from error
            if config.motion.type not in speeds:
                raise ConfigError(f"{path}: the service has no measured speed for a {config.motion.type!r} camera (jobs.seconds_per_video_second)")
            cameras[config.camera_id] = Camera(config, path, file_version(path))
        if not cameras:
            raise ConfigError(f"No camera file in {folder}")
        self._cameras = cameras

    def get(self, camera_id: str) -> Camera | None:
        return self._cameras.get(camera_id)

    @property
    def ids(self) -> list[str]:
        return list(self._cameras)

    def describe(self) -> list[dict[str, Any]]:
        return [camera.describe() for camera in self._cameras.values()]
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `274 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_service_cameras.py src/tvd/service/cameras.py
git commit -m "Check every camera file when the service starts"
```

### Task 3: A reference frame beyond the video, refused before any work

**Why:** Part 2 left this: a moving camera's `reference_frame` past the end of a video failed only after the whole pass, with a traceback. Now each motion estimator can say, from the frame count alone, whether it can use a video. The commands ask before loading a model, and the service will ask when a video is uploaded.

**Files:**
- Change: `tests/tvd/test_tvd_cli.py`
- Change: `tests/tvd/test_tvd_motion.py`
- Change: `src/tvd/motion/base.py`
- Change: `src/tvd/motion/orb.py`
- Change: `src/tvd/cli.py`

**Interfaces:**
- Consumes: `BaseMotionEstimator`, `OrbMotionEstimator` (Part 2); `run_check`, `run_review` (Parts 2 and 3).
- Produces: `BaseMotionEstimator.check_frames(frame_count)`, which raises `ValueError` if the video cannot be used, and does nothing by default; the ORB estimator's checks its reference frame.

- [ ] **Step 1: Write the failing test**

In `tests/tvd/test_tvd_cli.py`:

After:

```python
    assert json.loads(output.read_text())["rules"]["red_light"]["findings"][0]["trigger_frame"] == 70
```

add:

```python


def test_check_refuses_a_reference_frame_beyond_the_video_before_any_work(tmp_path, capsys, sample_video):
    (tmp_path / "camera.yaml").write_text(CAMERA_FILE.replace("frame_size: [640, 360]", "frame_size: [640, 360]\nmotion: {type: orb, options: {reference_frame: 500}}"))
    (tmp_path / "perception.yaml").write_text(DOWN)
    argv = ["check", "--video", str(sample_video), "--camera", str(tmp_path / "camera.yaml"), "--output", str(tmp_path / "f.json"), "--perception", str(tmp_path / "perception.yaml")]
    assert main(argv) == 2
    assert "error: The camera's reference frame 500 is beyond the video's 240 frames" in capsys.readouterr().err and not (tmp_path / "f.json").exists()
```

In `tests/tvd/test_tvd_motion.py`:

Before:

```python
def test_the_mask_hides_moving_objects_and_drawn_areas_only():
```

add:

```python
def test_a_video_too_short_for_the_reference_frame_is_refused_before_any_work():
    with pytest.raises(ValueError, match="The camera's reference frame 240 is beyond the video's 240 frames"):
        OrbMotionEstimator(reference_frame=240).check_frames(240)
    OrbMotionEstimator(reference_frame=240).check_frames(241)
    MOTION_ESTIMATORS.create("fixed").check_frames(1)
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_cli.py tests/tvd/test_tvd_motion.py -q
```

Expected: `2 failed, 16 passed`: `AttributeError: 'OrbMotionEstimator' object has no attribute 'check_frames'`, and the `check` command fails only after the pass, with `error: Reference frame 500 is outside the analysed frames 0 to 239`.

- [ ] **Step 3: Write the code**

In `src/tvd/motion/base.py`:

Before:

```python

MOTION_ESTIMATORS: Registry[BaseMotionEstimator] = Registry("camera motion estimator")
```

add:

```python
    def check_frames(self, frame_count: int) -> None:
        """Raise ``ValueError`` if a video of ``frame_count`` frames cannot be used, before any work. Any length can, by default."""
```

In `src/tvd/motion/orb.py`:

Before:

```python
    def static_mask(self, shape: tuple[int, int], detections: list[Detection]) -> np.ndarray:
```

add:

```python
    def check_frames(self, frame_count: int) -> None:
        if not 0 <= self.reference_frame < frame_count:
            raise ValueError(f"The camera's reference frame {self.reference_frame} is beyond the video's {frame_count} frames")
```

In `src/tvd/cli.py`:

After:

```python
    motion_estimator = build_motion_estimator(camera.motion)
    with VideoReader(video) as reader:
        check_frame_size(reader, camera)
```

add:

```python
        motion_estimator.check_frames(reader.frame_count)
```

After:

```python
    config_files = {name: file_version(path) for name, path in files.items()}
    with VideoReader(video) as reader:
        check_frame_size(reader, camera)
```

add:

```python
        motion_estimator.check_frames(reader.frame_count)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `276 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_cli.py tests/tvd/test_tvd_motion.py src/tvd/motion/base.py src/tvd/motion/orb.py src/tvd/cli.py
git commit -m "Refuse a reference frame beyond the video before any work"
```

### Task 4: Jobs and their states

**Why:** The spec keeps job states in a small SQLite table, so that a restart can still report them (Section 5). A state only moves forward, `queued`, `running`, then `done` or `failed`: the store refuses anything else. A job that was running when the service stopped cannot be finished, so it is marked failed when the service starts again, with the reason. Each call opens its own connection, because the API's threads and the worker's thread share the file.

**Files:**
- Create: `tests/tvd/test_tvd_jobs.py`
- Create: `src/tvd/service/jobs.py`

**Interfaces:**
- Consumes: Nothing from earlier tasks.
- Produces: `QUEUED`, `RUNNING`, `DONE`, `FAILED`, `INTERRUPTED`; `Job(job_id, camera_id, recorded_at, video_file, video_seconds, estimate_seconds, created_at, state, stage, frames_done, frames_total, started_at, finished_at, failure)` with `as_json()`; `JobStore(path)` with `add`, `get`, `next_queued`, `waiting`, `start`, `progress`, `finish`, `fail`, `interrupt`, `finished_before` and `delete`. Times are ISO 8601 text in UTC.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_jobs.py`:

```python
import pytest

from tvd.service.jobs import DONE, FAILED, INTERRUPTED, QUEUED, RUNNING, Job, JobStore


def job(job_id: str, created_at: str = "2026-11-07T10:00:00+00:00") -> Job:
    return Job(job_id, "cam1", "2026-11-07T17:00:00+07:00", "video.mp4", 60.0, 132.0, created_at)


def test_a_new_job_is_queued_and_read_back(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    assert store.get("a") == job("a") and store.get("a").state == QUEUED and store.get("missing") is None


def test_jobs_run_in_order_of_arrival(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("late", "2026-11-07T10:05:00+00:00"))
    store.add(job("early", "2026-11-07T10:01:00+00:00"))
    assert store.next_queued().job_id == "early" and [j.job_id for j in store.waiting()] == ["early", "late"]


def test_jobs_of_the_same_second_keep_the_order_they_were_added_in(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    for name in ("b", "c", "a"):
        store.add(job(name))
    assert [j.job_id for j in store.waiting()] == ["b", "c", "a"]


def test_a_job_runs_reports_its_progress_and_ends(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    store.start("a", "2026-11-07T10:01:00+00:00")
    store.progress("a", "tracking", 250, 1000)
    running = store.get("a")
    assert (running.state, running.stage, running.frames_done, running.frames_total) == (RUNNING, "tracking", 250, 1000) and store.next_queued() is None
    store.finish("a", "2026-11-07T10:03:00+00:00")
    assert (store.get("a").state, store.get("a").finished_at) == (DONE, "2026-11-07T10:03:00+00:00")


def test_a_failed_job_keeps_its_reason(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    store.start("a", "2026-11-07T10:01:00+00:00")
    store.fail("a", "RuntimeError: ffmpeg failed", "2026-11-07T10:02:00+00:00")
    assert (store.get("a").state, store.get("a").failure) == (FAILED, "RuntimeError: ffmpeg failed")


def test_a_state_only_moves_forward(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    with pytest.raises(ValueError, match="Job a is not running"):
        store.finish("a", "2026-11-07T10:01:00+00:00")
    store.start("a", "2026-11-07T10:01:00+00:00")
    with pytest.raises(ValueError, match="Job a is not queued"):
        store.start("a", "2026-11-07T10:02:00+00:00")


def test_after_a_restart_a_job_that_was_running_has_failed(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.add(job("a"))
    store.add(job("b", "2026-11-07T10:01:00+00:00"))
    store.start("a", "2026-11-07T10:01:00+00:00")
    restarted = JobStore(tmp_path / "jobs.sqlite")
    assert restarted.interrupt("2026-11-07T11:00:00+00:00") == ["a"]
    assert (restarted.get("a").state, restarted.get("a").failure, restarted.get("b").state) == (FAILED, INTERRUPTED, QUEUED)


def test_old_finished_jobs_can_be_found_and_deleted(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    for name in ("old", "new", "waiting"):
        store.add(job(name))
    for name, end in (("old", "2026-10-01T10:00:00+00:00"), ("new", "2026-11-07T10:00:00+00:00")):
        store.start(name, end)
        store.finish(name, end)
    assert [j.job_id for j in store.finished_before("2026-11-01T00:00:00+00:00")] == ["old"]
    store.delete("old")
    assert store.get("old") is None and store.get("waiting").state == QUEUED
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_jobs.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.jobs'`

- [ ] **Step 3: Write the code**

`src/tvd/service/jobs.py`:

```python
"""Jobs and their states, in a small SQLite table, so that a restart can still report every job.

A job is ``queued``, ``running``, ``done`` or ``failed``, and moves only forward. A job that was running
when the service stopped is marked ``failed`` when the service starts again, with the reason.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
INTERRUPTED = "The service stopped while this job was running"


@dataclass(frozen=True)
class Job:
    """Times are ISO 8601 in UTC. ``estimate_seconds`` is how long the job should take, from the measured speed."""

    job_id: str
    camera_id: str
    recorded_at: str
    video_file: str
    video_seconds: float
    estimate_seconds: float
    created_at: str
    state: str = QUEUED
    stage: str | None = None
    frames_done: int = 0
    frames_total: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    failure: str | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


COLUMNS = [f.name for f in fields(Job)]
SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, recorded_at TEXT NOT NULL, video_file TEXT NOT NULL,
    video_seconds REAL NOT NULL, estimate_seconds REAL NOT NULL, created_at TEXT NOT NULL, state TEXT NOT NULL,
    stage TEXT, frames_done INTEGER NOT NULL, frames_total INTEGER NOT NULL, started_at TEXT, finished_at TEXT, failure TEXT
)"""


class JobStore:
    """One connection per call, so that the API's threads and the worker's thread can share the file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:  # Commits at the end, or rolls back on an error.
                yield db
        finally:
            db.close()

    def _select(self, where: str, parameters: tuple = ()) -> list[Job]:
        """Jobs in order of arrival. Two that arrived in the same second keep the order in which they were added."""
        with self._connect() as db:
            rows = db.execute(f"SELECT {', '.join(COLUMNS)} FROM jobs WHERE {where} ORDER BY created_at, rowid", parameters).fetchall()
        return [Job(*row) for row in rows]

    def _update(self, job_id: str, states: tuple[str, ...], **values: Any) -> None:
        assignments = ", ".join(f"{name} = ?" for name in values)
        marks = ", ".join("?" for _ in states)
        with self._connect() as db:
            changed = db.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ? AND state IN ({marks})", (*values.values(), job_id, *states)).rowcount
        if changed != 1:
            raise ValueError(f"Job {job_id} is not {' or '.join(states)}")

    def add(self, job: Job) -> None:
        with self._connect() as db:
            db.execute(f"INSERT INTO jobs ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})", tuple(asdict(job).values()))

    def get(self, job_id: str) -> Job | None:
        found = self._select("job_id = ?", (job_id,))
        return found[0] if found else None

    def next_queued(self) -> Job | None:
        """The oldest queued job: jobs run in order of arrival."""
        found = self._select("state = ?", (QUEUED,))
        return found[0] if found else None

    def waiting(self) -> list[Job]:
        """The running job and the queued ones, in order."""
        return self._select("state IN (?, ?)", (RUNNING, QUEUED))

    def start(self, job_id: str, now: str) -> None:
        self._update(job_id, (QUEUED,), state=RUNNING, started_at=now)

    def progress(self, job_id: str, stage: str, frames_done: int, frames_total: int) -> None:
        self._update(job_id, (RUNNING,), stage=stage, frames_done=frames_done, frames_total=frames_total)

    def finish(self, job_id: str, now: str) -> None:
        self._update(job_id, (RUNNING,), state=DONE, finished_at=now)

    def fail(self, job_id: str, reason: str, now: str) -> None:
        self._update(job_id, (QUEUED, RUNNING), state=FAILED, failure=reason, finished_at=now)

    def interrupt(self, now: str) -> list[str]:
        """Mark every running job failed: the service has just started, so none of them is running any more."""
        running = [job.job_id for job in self._select("state = ?", (RUNNING,))]
        for job_id in running:
            self.fail(job_id, INTERRUPTED, now)
        return running

    def finished_before(self, moment: str) -> list[Job]:
        return self._select("state IN (?, ?) AND finished_at < ?", (DONE, FAILED, moment))

    def delete(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `284 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_jobs.py src/tvd/service/jobs.py
git commit -m "Keep jobs and their states in SQLite"
```

### Task 5: Each job's folder

**Why:** Each job has a folder, `<storage>/<job id>/` (spec, Section 6), which holds the uploaded video and, next to it, the review's files. So Part 3's review must accept a folder that already holds a file: it now refuses only a folder that holds an earlier review. **Only a job id leads to a folder,** and only a file inside it is ever served, so a path such as `../jobs.sqlite` gets nothing. A job id begins with the time of the upload, so ids sort by time.

**Files:**
- Change: `tests/tvd/test_tvd_cli.py`
- Change: `tests/tvd/test_tvd_review.py`
- Create: `tests/tvd/test_tvd_storage.py`
- Create: `src/tvd/service/storage.py`
- Change: `src/tvd/review.py`
- Change: `src/tvd/cli.py`

**Interfaces:**
- Consumes: `prepare_folder` (Part 3).
- Produces: `JobFolders(root)` with `database`, `new_id(now) -> str`, `folder(job_id) -> Path` (a `ValueError` for anything but a job id), `file(job_id, relative) -> Path | None` and `remove(job_id)`; `REVIEW_FILES`; `prepare_folder(folder)`, which now refuses only an earlier review or a file in the folder's place.

- [ ] **Step 1: Write the failing test**

Two tests of Part 3 change their expected message, and a third checks that a folder holding only the uploaded video is used.

In `tests/tvd/test_tvd_cli.py`:

Replace:

```python
    assert "error: The output must be a new or empty folder" in capsys.readouterr().err
```

with:

```python
    assert "error: The output folder already holds a review (records.json)" in capsys.readouterr().err
```

In `tests/tvd/test_tvd_review.py`:

Replace:

```python
    (tmp_path / "records.json").write_text("{}")
    with pytest.raises(FileExistsError, match="new or empty folder"):
```

with:

```python
    (tmp_path / "records.json").write_text("{}")
    with pytest.raises(FileExistsError, match=r"already holds a review \(records.json\)"):
```

Before:

```python
def test_a_file_where_the_folder_should_be_is_refused(tmp_path, sample_video):
```

add:

```python
def test_a_folder_that_holds_only_the_uploaded_video_is_used(tmp_path, sample_video):
    (tmp_path / "video.mp4").write_bytes(sample_video.read_bytes())
    assert len(review(tmp_path / "video.mp4", tmp_path, FakePlateModels()).records) == 2 and (tmp_path / "records.json").is_file()
```

Replace:

```python
    with pytest.raises(FileExistsError, match="new or empty folder"):
```

with:

```python
    with pytest.raises(FileExistsError, match="must be a folder"):
```

`tests/tvd/test_tvd_storage.py`:

```python
import os
from datetime import datetime, timezone

import pytest

from tvd.service.storage import JobFolders

NOW = datetime(2026, 11, 7, 10, 30, 5, tzinfo=timezone.utc)


def test_a_job_id_starts_with_the_upload_time(tmp_path):
    folders = JobFolders(tmp_path / "storage")
    first, second = folders.new_id(NOW), folders.new_id(NOW)
    assert first.startswith("20261107-103005-") and len(first) == 22 and first != second
    assert folders.folder(first) == tmp_path / "storage" / first and folders.database == tmp_path / "storage" / "jobs.sqlite"


@pytest.mark.parametrize("job_id", ["..", "../storage", "20261107-103005-abc", "20261107-103005-abcdef/.."])
def test_anything_but_a_job_id_is_refused(tmp_path, job_id):
    with pytest.raises(ValueError, match="Not a job id"):
        JobFolders(tmp_path).folder(job_id)


def test_only_files_inside_the_job_folder_are_given(tmp_path):
    folders = JobFolders(tmp_path / "storage")
    job_id = folders.new_id(NOW)
    folder = folders.folder(job_id)
    (folder / "evidence").mkdir(parents=True)
    (folder / "evidence" / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "secret.txt").write_text("not for the backend")
    os.symlink(tmp_path / "secret.txt", folder / "evidence" / "link.jpg")
    assert folders.file(job_id, "evidence/a.jpg") == (folder / "evidence" / "a.jpg").resolve()
    assert [folders.file(job_id, p) for p in ("../../secret.txt", "evidence/link.jpg", "evidence", "missing.jpg")] == [None] * 4


def test_a_job_folder_is_removed(tmp_path):
    folders = JobFolders(tmp_path)
    job_id = folders.new_id(NOW)
    folders.folder(job_id).mkdir()
    folders.remove(job_id)
    assert not folders.folder(job_id).exists()
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_cli.py tests/tvd/test_tvd_review.py tests/tvd/test_tvd_storage.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.storage'`. pytest stops there; alone, the new test of `test_tvd_review.py` fails with `FileExistsError: The output must be a new or empty folder`.

- [ ] **Step 3: Write the code**

`src/tvd/service/storage.py`:

```python
"""Each job's folder, ``<storage>/<job id>/``: the uploaded video, then the review's files next to it."""

from __future__ import annotations

import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path

JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


class JobFolders:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    @property
    def database(self) -> Path:
        return self.root / "jobs.sqlite"

    def new_id(self, now: datetime) -> str:
        """The time of the upload, and 6 random hex digits: ids sort by time and never repeat in practice."""
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def folder(self, job_id: str) -> Path:
        """A job's folder. An id that is not one, such as ``../x``, is refused, so it never leads outside the storage."""
        if not JOB_ID.match(job_id):
            raise ValueError(f"Not a job id: {job_id!r}")
        return self.root / job_id

    def file(self, job_id: str, relative: str) -> Path | None:
        """A file inside a job's folder, or None. A path that leads outside the folder, even through a link, is never given."""
        folder = self.folder(job_id).resolve()
        path = (folder / relative).resolve()
        return path if path.is_relative_to(folder) and path.is_file() else None

    def remove(self, job_id: str) -> None:
        shutil.rmtree(self.folder(job_id), ignore_errors=True)
```

In `src/tvd/review.py`:

Before:

```python
def prepare_folder(folder: Path) -> None:
```

add:

```python
REVIEW_FILES = ("records.json", "summary.json", "clips", "evidence", "plates")
```

Replace:

```python
    """A new folder, or an empty one: files of an earlier run could show the wrong moment."""
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError(f"The output must be a new or empty folder: {folder}")
```

with:

```python
    """A folder with no earlier review, whose files could show the wrong moment. Other files, such as the uploaded video, may be there."""
    if folder.exists() and not folder.is_dir():
        raise FileExistsError(f"The output must be a folder: {folder}")
    earlier = [name for name in REVIEW_FILES if (folder / name).exists()]
    if earlier:
        raise FileExistsError(f"The output folder already holds a review ({', '.join(earlier)}): {folder}")
```

In `src/tvd/cli.py`:

Replace:

```python
    review.add_argument("--output", type=Path, required=True, help="A new or empty folder")
```

with:

```python
    review.add_argument("--output", type=Path, required=True, help="A folder with no earlier review")
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `292 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_cli.py tests/tvd/test_tvd_review.py tests/tvd/test_tvd_storage.py src/tvd/service/storage.py src/tvd/review.py src/tvd/cli.py
git commit -m "Give each job a folder, and let a review share it with the upload"
```

### Task 6: An upload, checked before it becomes a job

**Why:** The spec's upload checks (Section 5): the file is a readable video of that camera's frame size, and not too long for the deadline. Anything refused here is never queued, and says why, with the HTTP status the API will answer: 413 for an upload larger than the limit, 422 for the rest. The upload is copied in pieces, and a larger one is refused while it arrives.

**Files:**
- Create: `tests/tvd/test_tvd_intake.py`
- Create: `src/tvd/service/intake.py`

**Interfaces:**
- Consumes: `check_frame_size`, `VideoReader` (Parts 1 and 2); `check_frames` (Task 3); `Camera` (Task 2); `JobSettings` (Task 1).
- Produces: `Refused(reason, status=422)`; `save_upload(source, path, max_bytes) -> int`; `VideoFacts(frame_count, fps, width, height)` with `seconds`; `inspect_video(path, camera) -> VideoFacts`; `estimate_seconds(facts, camera, jobs) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_intake.py`:

```python
import io

import pytest
from conftest import ROOT

from tvd.camera import CameraConfig
from tvd.service.cameras import Camera
from tvd.service.intake import Refused, estimate_seconds, inspect_video, save_upload
from tvd.service.settings import JobSettings

SCENE = {"stop_line": [[0, 150], [640, 150]], "lane_dividers": [0, 600], "lane_direction_deg": 90, "traffic_direction": "toward_camera",
         "signals": {"reader": "test-fixed-state", "through": [{"state": "red"}]}}
JOBS = JobSettings(deadline_minutes=120, seconds_per_video_second={"fixed": 2.2, "orb": 7.0})


def camera(frame_size=(640, 360), motion=None) -> Camera:
    config = {"camera_id": "test-camera", "frame_size": list(frame_size), "rules": {"red_light": {"scene": SCENE}}}
    if motion:
        config["motion"] = motion
    return Camera(CameraConfig.model_validate(config), ROOT / "test-camera.yaml", "sha256:000000000000")


def test_an_upload_is_saved_whole(tmp_path):
    assert save_upload(io.BytesIO(b"x" * 3_000_000), tmp_path / "job" / "video.mp4", 4_000_000) == 3_000_000
    assert (tmp_path / "job" / "video.mp4").stat().st_size == 3_000_000


def test_a_larger_upload_is_refused_and_nothing_is_kept(tmp_path):
    with pytest.raises(Refused, match="larger than 2 MB") as refused:
        save_upload(io.BytesIO(b"x" * 3_000_000), tmp_path / "video.mp4", 2 * 2**20)
    assert refused.value.status == 413 and not (tmp_path / "video.mp4").exists()


def test_an_empty_upload_is_refused(tmp_path):
    with pytest.raises(Refused, match="empty"):
        save_upload(io.BytesIO(b""), tmp_path / "video.mp4", 100)


def test_a_video_of_the_camera_is_described(sample_video):
    facts = inspect_video(sample_video, camera())
    assert (facts.frame_count, facts.fps, facts.width, facts.height, facts.seconds) == (240, 24.0, 640, 360, 10.0)


@pytest.mark.parametrize("change, reason", [
    ({"frame_size": (1920, 1080)}, "Camera test-camera has 1920x1080 frames, but traffic_sample.mp4 is 640x360"),
    ({"motion": {"type": "orb", "options": {"reference_frame": 500}}}, "reference frame 500 is beyond the video's 240 frames"),
])
def test_a_video_that_cannot_be_reviewed_with_this_camera_is_refused(sample_video, change, reason):
    with pytest.raises(Refused, match=reason) as refused:
        inspect_video(sample_video, camera(**change))
    assert refused.value.status == 422


def test_a_file_that_is_not_a_video_is_refused(tmp_path):
    (tmp_path / "video.mp4").write_text("not a video")
    with pytest.raises(Refused, match="could not open"):
        inspect_video(tmp_path / "video.mp4", camera())


def test_the_estimate_follows_the_measured_speed_of_the_camera(sample_video):
    facts = inspect_video(sample_video, camera())
    assert estimate_seconds(facts, camera(), JOBS) == pytest.approx(22.0)
    assert estimate_seconds(facts, camera(motion={"type": "orb", "options": {"reference_frame": 0}}), JOBS) == pytest.approx(70.0)


def test_a_video_too_long_for_the_deadline_is_refused(sample_video):
    short_deadline = JobSettings(deadline_minutes=0.3, seconds_per_video_second={"fixed": 2.2})
    with pytest.raises(Refused, match="lasts 0.2 min, and a video of a fixed camera can last 0.1 min at most, to be reviewed within 0.3 min"):
        estimate_seconds(inspect_video(sample_video, camera()), camera(), short_deadline)
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_intake.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.intake'`

- [ ] **Step 3: Write the code**

`src/tvd/service/intake.py`:

```python
"""What an upload must be before it becomes a job. An upload refused here is never queued, and says why.

``Refused.status`` is the HTTP status that the API answers with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from tvd.check import check_frame_size
from tvd.config import ConfigError
from tvd.factory import build_motion_estimator
from tvd.service.cameras import Camera
from tvd.service.settings import JobSettings
from tvd.video import VideoReader

CHUNK_BYTES = 1 << 20


class Refused(Exception):
    def __init__(self, reason: str, status: int = 422) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def save_upload(source: BinaryIO, path: Path, max_bytes: int) -> int:
    """Copy an upload to ``path`` in pieces, and return its size. A larger one is refused while it arrives, and nothing of it is kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with path.open("wb") as out:
            while chunk := source.read(CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise Refused(f"The upload is larger than {max_bytes / 2**20:.0f} MB", 413)
                out.write(chunk)
        if written == 0:
            raise Refused("The upload is empty")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return written


@dataclass(frozen=True)
class VideoFacts:
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def seconds(self) -> float:
        return self.frame_count / self.fps


def inspect_video(path: Path, camera: Camera) -> VideoFacts:
    """The facts of an uploaded video, if it is a video of this camera that can be reviewed."""
    try:
        with VideoReader(path) as reader:
            check_frame_size(reader, camera.config)
            facts = VideoFacts(reader.frame_count, reader.fps, reader.width, reader.height)
    except (RuntimeError, ConfigError) as error:  # Not a video, or a video of another camera's frame size.
        raise Refused(str(error)) from error
    if facts.frame_count <= 0:
        raise Refused(f"{path.name} has no frames")
    try:
        build_motion_estimator(camera.config.motion).check_frames(facts.frame_count)
    except ValueError as error:
        raise Refused(str(error)) from error
    return facts


def estimate_seconds(facts: VideoFacts, camera: Camera, jobs: JobSettings) -> float:
    """How long the job should take. A video too long to be done within the deadline, even with no queue, is refused."""
    longest = jobs.longest_video_seconds(camera.motion_type)
    if facts.seconds > longest:
        raise Refused(
            f"The video lasts {facts.seconds / 60:.1f} min, and a video of a {camera.motion_type} camera can last "
            f"{longest / 60:.1f} min at most, to be reviewed within {jobs.deadline_minutes:g} min"
        )
    return jobs.work_seconds(camera.motion_type, facts.seconds)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `301 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_intake.py src/tvd/service/intake.py
git commit -m "Check an upload before it becomes a job"
```

### Task 7: How far a job has gone

**Why:** `GET /jobs/{id}` must show progress (spec, Section 5). The pass over the video already feeds observers, so progress is one more: a `FrameCounter` reports how many frames it has seen. `check_video` only learns to take extra observers; it does not know what they do (**open/closed**).

**Files:**
- Change: `tests/tvd/test_tvd_check.py`
- Change: `tests/tvd/test_tvd_perception.py`
- Change: `src/tvd/observers.py`
- Change: `src/tvd/check.py`

**Interfaces:**
- Consumes: `FrameObserver` (Part 2); `check_video`, `track_video` (Parts 1 and 2).
- Produces: `FrameCounter(report, every, total)`, which calls `report(count)` every `every` frames and at frame `total`; `check_video(..., observers=())`.

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/tvd/test_tvd_check.py`:

```python
def test_more_observers_see_every_frame_of_the_pass(sample_video):
    from tvd.observers import FrameCounter

    reported = []
    config = CameraConfig.model_validate(CAMERA)
    with VideoReader(sample_video) as reader:
        check_video(
            reader, config, build_detector(ComponentConfig(type="test-moving-car", options={"step": 0, "step_y": 1})), build_tracker(ComponentConfig(type="test-same-id")),
            {"car"}, build_motion_estimator(config.motion), build_rules(config, DEFAULTS), 40, observers=[FrameCounter(reported.append, every=50, total=200)],
        )
    assert reported == [50, 100, 150, 200]
```

In `tests/tvd/test_tvd_perception.py`:

Replace:

```python
from tvd.observers import DetectionLog, FrameObserver
```

with:

```python
from tvd.observers import DetectionLog, FrameCounter, FrameObserver
```

Add to the end of the file:

```python
def test_a_frame_counter_reports_every_n_frames_and_the_last(sample_video):
    reported = []
    counter = FrameCounter(reported.append, every=100, total=240)
    with VideoReader(sample_video) as reader:
        track_video(reader, DETECTORS.create("test-moving-car"), TRACKERS.create("test-same-id"), {"car"}, observers=[counter])
    assert reported == [100, 200, 240] and counter.count == 240
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_check.py tests/tvd/test_tvd_perception.py -q
```

Expected: `E   ImportError: cannot import name 'FrameCounter' from 'tvd.observers'`

- [ ] **Step 3: Write the code**

In `src/tvd/observers.py`:

After:

```python
from abc import ABC, abstractmethod
```

add:

```python
from collections.abc import Callable
```

Add to the end of the file:

```python
class FrameCounter(FrameObserver):
    """Counts the frames of the pass, and reports the count every ``every`` frames and at the last one: a job's progress."""

    def __init__(self, report: Callable[[int], None], every: int, total: int) -> None:
        self.report = report
        self.every = every
        self.total = total
        self.count = 0

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self.count += 1
        if self.count % self.every == 0 or self.count == self.total:
            self.report(self.count)
```

In `src/tvd/check.py`:

After:

```python
from __future__ import annotations

import time
```

add:

```python
from collections.abc import Sequence
```

Replace:

```python
from tvd.observers import DetectionLog
```

with:

```python
from tvd.observers import DetectionLog, FrameObserver
```

After:

```python
    start: int = 0,
    stop: int | None = None,
```

add:

```python
    observers: Sequence[FrameObserver] = (),
```

Replace:

```python
    """Detect and track once, and give every rule the same tracks. Build new rules for each video: their observers keep what they saw."""
```

with:

```python
    """Detect and track once, and give every rule the same tracks. Build new rules for each video: their observers keep what they saw.

    ``observers`` see every frame too, as the rules' own do: a job's progress, for example.
    """
```

Replace:

```python
    observers = [o for rule in rules for o in rule.frame_observers()] + ([log] if log else [])
    history = track_video(reader, detector, tracker, track_classes, start, stop, observers)
```

with:

```python
    everyone = [o for rule in rules for o in rule.frame_observers()] + ([log] if log else []) + list(observers)
    history = track_video(reader, detector, tracker, track_classes, start, stop, everyone)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `303 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_check.py tests/tvd/test_tvd_perception.py src/tvd/observers.py src/tvd/check.py
git commit -m "Report how far the pass has gone"
```

### Task 8: Plate models that do not answer in time

**Why:** Part 3 left this: a call to the plate models' process had no time limit, so a model that hung would hold the queue for ever. Now a call that is not answered within `timeout_seconds` stops the process, and the candidate's record says that its plate was not read. The models then say they are not alive, and the service starts them again for the next job (Task 9).

**Files:**
- Change: `tests/tvd/test_tvd_onnx_plates.py`
- Change: `tests/tvd/test_tvd_plates.py`
- Change: `configs/tvd/plates.yaml`
- Change: `src/tvd/plates/base.py`
- Change: `src/tvd/plates/onnx.py`

**Interfaces:**
- Consumes: `OnnxPlateModels`, `BasePlateModels` (Part 3).
- Produces: `BasePlateModels.is_alive() -> bool`, True by default; `OnnxPlateModels(detector_model, ocr_model, cpu_threads, timeout_seconds)`; `timeout_seconds: 60` in `configs/tvd/plates.yaml`.

- [ ] **Step 1: Write the failing test**

The test of the time limit gives the models 0.1 ms to find plates in a 1080p frame: they never make it.

In `tests/tvd/test_tvd_onnx_plates.py`:

Before:

```python
def test_a_stopped_process_gives_a_clear_error():
```

add:

```python
def test_models_that_do_not_answer_in_time_are_stopped():
    hasty = build_plate_models(PLATES.models.model_copy(update={"options": {**PLATES.models.options, "timeout_seconds": 0.0001}}))
    try:
        assert hasty.is_alive()
        with pytest.raises(RuntimeError, match="did not answer within 0.0001 s"):
            hasty.detect(np.zeros((1080, 1920, 3), np.uint8))
        assert not hasty.is_alive()
    finally:
        hasty.close()


def test_closed_models_are_not_alive():
    started = build_plate_models(PLATES.models)
    started.close()
    assert not started.is_alive()
```

In `tests/tvd/test_tvd_plates.py`:

Replace:

```python
    assert {"version": PLATES.version, **PLATES.models.options, **READING.model_dump()} == PLATE_SETTINGS and PLATES.models.type == "onnx_worker"
```

with:

```python
    options = {k: v for k, v in PLATES.models.options.items() if k != "timeout_seconds"}  # New in the service: the notebooks wait for ever.
    assert {"version": PLATES.version, **options, **READING.model_dump()} == PLATE_SETTINGS and PLATES.models.type == "onnx_worker"
    assert PLATES.models.options["timeout_seconds"] == 60
```

After:

```python
        assert models.recognize(IMAGE).characters == "FAKE9999_"
    assert models.closed
```

add:

```python


def test_models_without_a_process_are_always_alive():
    assert FakePlateModels().is_alive()
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_onnx_plates.py tests/tvd/test_tvd_plates.py -q
```

Expected: `4 failed, 38 passed`: `AttributeError: 'OnnxPlateModels' object has no attribute 'is_alive'`, `KeyError: 'timeout_seconds'` and `AttributeError: 'FakePlateModels' object has no attribute 'is_alive'`.

- [ ] **Step 3: Write the code**

In `configs/tvd/plates.yaml`:

After:

```yaml
    cpu_threads: 8            # only for what ONNX runs on the CPU, such as on a machine with no GPU
```

add:

```yaml
    timeout_seconds: 60       # a call not answered within this stops the models' process: one frame takes milliseconds
```

In `src/tvd/plates/base.py`:

Before:

```python
    def __enter__(self) -> BasePlateModels:
```

add:

```python
    def is_alive(self) -> bool:
        """False once the models can no longer answer, and must be started again. Always True, by default."""
        return True
```

In `src/tvd/plates/onnx.py`:

After:

```python
import json
import os
import pickle
```

add:

```python
import select
```

Replace:

```python
    def __init__(self, detector_model: str, ocr_model: str, cpu_threads: int) -> None:
```

with:

```python
    def __init__(self, detector_model: str, ocr_model: str, cpu_threads: int, timeout_seconds: float) -> None:
```

After:

```python
        options = {"detector_model": detector_model, "ocr_model": ocr_model, "cpu_threads": cpu_threads}
```

add:

```python
        self.timeout_seconds = timeout_seconds
```

Replace:

```python
        self.providers: list[str] = self._receive()
```

with:

```python
        self.providers: list[str] = self._receive()  # Loading the models may take long, the first time: no time limit.
```

Replace:

```python
        return self._process.pid

    def _receive(self) -> Any:
```

with:

```python
        return self._process.pid

    def _receive(self, timeout: float | None = None) -> Any:
        if timeout is not None and not select.select([self._process.stdout], [], [], timeout)[0]:
            self._process.kill()
            self._process.wait()
            raise RuntimeError(f"The plate models did not answer within {timeout:g} s; their process was stopped")
```

Replace:

```python
            raise RuntimeError(f"The plate models' process stopped (exit code {self._process.poll()}).") from error
        return self._receive()
```

with:

```python
            raise RuntimeError(f"The plate models' process stopped (exit code {self._process.poll()}).") from error
        return self._receive(self.timeout_seconds)
```

Add to the end of the file:

```python
    def is_alive(self) -> bool:
        return self._process.poll() is None
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `306 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_onnx_plates.py tests/tvd/test_tvd_plates.py configs/tvd/plates.yaml src/tvd/plates/base.py src/tvd/plates/onnx.py
git commit -m "Stop plate models that do not answer in time"
```

### Task 9: Models loaded once, and one job

**Why:** The detector, the tracker and the plate models take seconds to load, so they load once, when the service starts (spec, Section 5). A job's rules and motion estimator are built new for it, from its camera, because they keep what they saw of one video. A job's records use its id where the command used the video's name, so two uploads with the same file name never share an event id.

**Files:**
- Create: `tests/tvd/service_data.py`
- Create: `tests/tvd/test_tvd_runner.py`
- Create: `src/tvd/service/runner.py`

**Interfaces:**
- Consumes: `check_video` (Task 7), `review_check` (Part 3), `FrameCounter` (Task 7), `Camera` (Task 2), `Job` (Task 4), `is_alive` (Task 8), `SettingsFiles` (Task 1).
- Produces: `Pipeline(files)` with `plate_models()` (started again when gone), `plates_unavailable`, `run(job, camera, folder, report) -> ReviewResult` and `close()`. `report(stage, frames_done, frames_total)`, with the stages `tracking`, then `camera motion` or `rules`, then `evidence`. `tests/tvd/service_data.py` writes the settings files of a test service.

- [ ] **Step 1: Write the failing test**

`tests/tvd/service_data.py`:

```python
"""Settings files and a camera for service tests: the test detector and tracker, fake plate models, and the committed sample video."""

import shutil
from pathlib import Path

from conftest import ROOT

from tvd.service.settings import load_service_settings

PERCEPTION = """
detector:
  type: test-moving-car
  options: {step: 0, step_y: 1}
tracker:
  type: test-same-id
track_classes: [car]
"""
PLATES = """
version: 1
models:
  type: test-plate
reading: {frames_per_track: 12, min_frame_gap: 4, vehicle_margin: 0.15, lowest_plate_center: 0.4, min_plate_width: 30, min_chars: 4, min_char_prob: 0.95, min_votes: 2, min_agreement: 0.67}
"""
# The test car comes down the image by 1 px per frame: through a red light, against the only allowed direction.
CAMERA = """
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
  wrong_way:
    scene:
      vanishing_point: [320, -100000]
      regions: [{id: road, polygon: [[0, 0], [640, 0], [640, 360], [0, 360]], allowed_direction: away_from_camera, tolerance_deg: 60}]
"""


def service_files(tmp_path: Path, plates: str = PLATES) -> Path:
    """A service.yaml for tests, in tmp_path, with its own storage, cameras, perception and plate files."""
    (tmp_path / "cameras").mkdir()
    (tmp_path / "cameras" / "test-camera.yaml").write_text(CAMERA)
    (tmp_path / "perception.yaml").write_text(PERCEPTION)
    (tmp_path / "plates.yaml").write_text(plates)
    text = (ROOT / "configs" / "service.yaml").read_text()
    for old, new in (
        ("outputs/service", str(tmp_path / "storage")), ("configs/cameras", str(tmp_path / "cameras")),
        ("configs/tvd/perception.yaml", str(tmp_path / "perception.yaml")), ("configs/tvd/plates.yaml", str(tmp_path / "plates.yaml")),
        ("configs/tvd/rules.yaml", str(ROOT / "configs" / "tvd" / "rules.yaml")), ("configs/tvd/evidence.yaml", str(ROOT / "configs" / "tvd" / "evidence.yaml")),
    ):
        text = text.replace(old, new)
    (tmp_path / "service.yaml").write_text(text)
    return tmp_path / "service.yaml"


def settings_for(tmp_path: Path, plates: str = PLATES):
    return load_service_settings(service_files(tmp_path, plates))


def copy_video(source: Path, folder: Path, name: str = "video.mp4") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / name)
    return folder / name
```

`tests/tvd/test_tvd_runner.py`:

```python
import json

from conftest import FakePlateModels
from service_data import PLATES, copy_video, settings_for

from tvd.camera import load_rule_defaults
from tvd.plates import PLATE_MODELS, BasePlateModels
from tvd.service.cameras import CameraCatalog
from tvd.service.jobs import Job
from tvd.service.runner import Pipeline

JOB_ID = "20261107-103005-abcdef"


@PLATE_MODELS.register("test-one-use-plate")
class OneUse(FakePlateModels):
    """Models that are gone after they started, as after a call that did not answer in time."""

    def is_alive(self) -> bool:
        return False


@PLATE_MODELS.register("test-never-starts")
class NeverStarts(BasePlateModels):
    def __init__(self) -> None:
        raise RuntimeError("Plate models: no such model")

    def detect(self, image):
        return []

    def recognize(self, plate):
        raise NotImplementedError


def setup(tmp_path, sample_video, plates=PLATES):
    settings = settings_for(tmp_path, plates)
    camera = CameraCatalog(settings.files.cameras, load_rule_defaults(settings.files.rules), settings.jobs.seconds_per_video_second).get("test-camera")
    folder = tmp_path / "storage" / JOB_ID
    copy_video(sample_video, folder)
    job = Job(JOB_ID, "test-camera", "2026-11-07T08:00:00+07:00", "video.mp4", 10.0, 22.0, "2026-11-07T01:00:00+00:00")
    return Pipeline(settings.files), job, camera, folder


def test_a_job_is_reviewed_in_its_folder_with_the_job_id_in_every_record(tmp_path, sample_video):
    pipeline, job, camera, folder = setup(tmp_path, sample_video)
    reports = []
    result = pipeline.run(job, camera, folder, lambda *report: reports.append(report))
    records = json.loads((folder / "records.json").read_text())["events"]
    assert [r["event_id"] for r in records] == [f"evt-{JOB_ID}-wrong_way-f00048-t1", f"evt-{JOB_ID}-red_light-f00070-t1"] and len(result.records) == 2
    assert records[0]["track"]["tracking_session"] == f"{JOB_ID}/test-same-id" and records[0]["config_files"]["camera"] == camera.version
    assert set(records[0]["config_files"]) == {"perception", "rules", "plates", "evidence", "camera"} and (folder / "video.mp4").is_file()
    assert reports[0] == ("tracking", 0, 240) and reports[1] == ("tracking", 25, 240) and reports[-2:] == [("rules", 240, 240), ("evidence", 240, 240)]


def test_plate_models_that_are_gone_are_started_again(tmp_path, sample_video):
    pipeline, *_ = setup(tmp_path, sample_video, PLATES.replace("type: test-plate", "type: test-one-use-plate"))
    first = pipeline.plate_models()
    second = pipeline.plate_models()
    assert first is not second and first.closed


def test_without_plate_models_every_record_says_why(tmp_path, sample_video):
    pipeline, job, camera, folder = setup(tmp_path, sample_video, PLATES.replace("type: test-plate", "type: test-never-starts"))
    result = pipeline.run(job, camera, folder, lambda *report: None)
    assert result.failures == ["plate reading: the plate models did not start: Plate models: no such model"]
    assert {r["license_plate"]["status"] for r in result.records} == {"not_attempted"}
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_runner.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.runner'`

- [ ] **Step 3: Write the code**

`src/tvd/service/runner.py`:

```python
"""What is loaded once for every job, and how one job is done: Part 2's pass, then Part 3's review."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tvd.camera import load_rule_defaults
from tvd.check import check_video
from tvd.config import PerceptionConfig, file_version, load_config
from tvd.evidence import load_evidence_settings
from tvd.factory import build_detector, build_motion_estimator, build_plate_models, build_rules, build_tracker
from tvd.observers import FrameCounter
from tvd.plates import BasePlateModels, load_plates_config
from tvd.records import RecordContext, parse_recorded_at
from tvd.review import ReviewResult, review_check
from tvd.service.cameras import Camera
from tvd.service.jobs import Job
from tvd.service.settings import SettingsFiles
from tvd.video import VideoReader

PROGRESS_EVERY = 25  # frames between two progress reports

Report = Callable[[str, int, int], None]
"""``report(stage, frames_done, frames_total)``: the stages are tracking, camera motion or rules, then evidence."""


class Pipeline:
    """The detector, the tracker and the plate models, loaded once when the service starts, with the settings of every job.

    Each job gets new rules and a new motion estimator, built from its camera: they keep what they saw of one video.
    """

    def __init__(self, files: SettingsFiles) -> None:
        self.perception = load_config(files.perception, PerceptionConfig)
        self.rule_defaults = load_rule_defaults(files.rules)
        self.plates = load_plates_config(files.plates)
        self.evidence = load_evidence_settings(files.evidence)
        self.config_files = {name: file_version(getattr(files, name)) for name in ("perception", "rules", "plates", "evidence")}
        self.detector = build_detector(self.perception.detector)
        self.tracker = build_tracker(self.perception.tracker)
        self._plate_models: BasePlateModels | None = None
        self.plates_unavailable = "no plate models"
        self._start_plate_models()

    def _start_plate_models(self) -> None:
        try:
            self._plate_models = build_plate_models(self.plates.models)
        except RuntimeError as error:  # The review goes on without plates, and its records say why.
            self._plate_models, self.plates_unavailable = None, f"the plate models did not start: {error}"

    def plate_models(self) -> BasePlateModels | None:
        """The plate models, started again if they are gone, as after a call that did not answer in time."""
        if self._plate_models is None or not self._plate_models.is_alive():
            if self._plate_models is not None:
                self._plate_models.close()
            self._start_plate_models()
        return self._plate_models

    def run(self, job: Job, camera: Camera, folder: Path, report: Report) -> ReviewResult:
        """Review the job's video, which is in ``folder``, and write the review there."""
        with VideoReader(folder / job.video_file) as reader:
            total = reader.frame_count
            after_tracking = "camera motion" if camera.motion_type != "fixed" else "rules"
            counter = FrameCounter(lambda done: report("tracking" if done < total else after_tracking, done, total), PROGRESS_EVERY, total)
            report("tracking", 0, total)
            check = check_video(
                reader, camera.config, self.detector, self.tracker, set(self.perception.track_classes),
                build_motion_estimator(camera.config.motion), build_rules(camera.config, self.rule_defaults), observers=[counter],
            )
            report("evidence", check.frames, total)
            settings = {
                "perception": self.perception.model_dump(), "rules": {name: r.settings for name, r in check.results.items()},
                "plates": self.plates.model_dump(), "evidence": self.evidence.model_dump(),
            }
            context = RecordContext(
                job.job_id, reader.fps, reader.width, reader.height, camera.camera_id, parse_recorded_at(job.recorded_at),
                f"{job.job_id}/{self.perception.tracker.type}", settings, {**self.config_files, "camera": camera.version},
            )
            return review_check(reader, check, context, self.plate_models(), self.plates.reading, self.evidence, folder, self.plates_unavailable)

    def close(self) -> None:
        if self._plate_models is not None:
            self._plate_models.close()
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `309 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/service_data.py tests/tvd/test_tvd_runner.py src/tvd/service/runner.py
git commit -m "Load the models once, and run one job with them"
```

### Task 10: The service's core

**Why:** The spec asks for a core Python API with a thin REST layer over it (Section 8), so that everything can be tested without HTTP, and a queue could replace REST later. `ReviewService` checks an upload (Task 6), refuses it if it would miss the deadline behind the jobs already waiting, and queues it; `run_next` runs the oldest job. **One lock** keeps two uploads from both taking the last room in the queue, and **a lock file** keeps a second service from running the same queue.

**Files:**
- Change: `tests/tvd/service_data.py`
- Create: `tests/tvd/test_tvd_service.py`
- Create: `src/tvd/service/core.py`

**Interfaces:**
- Consumes: Tasks 1 to 9.
- Produces: `ReviewService(settings, clock=utc_now, pipeline=None)` with `submit(upload, filename, camera_id, recorded_at) -> Job`, `queue_seconds()`, `status(job_id) -> Job`, `records(job_id) -> dict`, `file(job_id, path) -> Path`, `describe_cameras()`, `run_next() -> Job | None`, `remove_old() -> list[str]`, `interrupted` and `close()`; `NotFound`, `NotReady`.

- [ ] **Step 1: Write the failing test**

The tests give the service a clock of their own, so that the retention can be tested 8 days later without waiting.

In `tests/tvd/service_data.py`:

Replace:

```python
    """A service.yaml for tests, in tmp_path, with its own storage, cameras, perception and plate files."""
    (tmp_path / "cameras").mkdir()
```

with:

```python
    """A service.yaml for tests, in tmp_path, with its own storage, cameras, perception and plate files."""
    (tmp_path / "cameras").mkdir(exist_ok=True)
```

`tests/tvd/test_tvd_service.py`:

```python
import io
from datetime import datetime, timedelta, timezone

import pytest
from service_data import settings_for

from tvd.service.core import NotFound, NotReady, ReviewService
from tvd.service.intake import Refused
from tvd.service.jobs import DONE, FAILED, INTERRUPTED, QUEUED

RECORDED_AT = "2026-11-07T08:00:00+07:00"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 7, 1, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def start(tmp_path, clock=None, **changes):
    settings = settings_for(tmp_path)
    if changes:
        settings = settings.model_copy(update={"jobs": settings.jobs.model_copy(update=changes)})
    return ReviewService(settings, clock or Clock())


def submit(service, video, camera_id="test-camera", recorded_at=RECORDED_AT, name="cam1 08h00.mp4"):
    with open(video, "rb") as upload:
        return service.submit(upload, name, camera_id, recorded_at)


def test_an_upload_becomes_a_queued_job_with_its_video_in_its_folder(tmp_path, sample_video):
    service = start(tmp_path)
    job = submit(service, sample_video)
    assert (job.state, job.camera_id, job.video_file, job.video_seconds, job.estimate_seconds) == (QUEUED, "test-camera", "video.mp4", 10.0, 22.0)
    assert job.job_id.startswith("20261107-010000-") and service.status(job.job_id) == job
    assert (service.folders.folder(job.job_id) / "video.mp4").read_bytes() == sample_video.read_bytes()


@pytest.mark.parametrize("change, reason, status", [
    ({"recorded_at": "2026-11-07T08:00:00"}, "needs a time zone", 422),
    ({"camera_id": "cam9"}, "Unknown camera 'cam9'. Known cameras: test-camera", 422),
])
def test_a_refused_upload_leaves_nothing_behind(tmp_path, sample_video, change, reason, status):
    service = start(tmp_path)
    with pytest.raises(Refused, match=reason) as refused:
        submit(service, sample_video, **change)
    assert refused.value.status == status and service.store.waiting() == [] and sorted(p.name for p in service.folders.root.iterdir()) == ["jobs.sqlite", "service.lock"]


def test_a_file_that_is_not_a_video_leaves_nothing_behind(tmp_path):
    service = start(tmp_path)
    with pytest.raises(Refused, match="could not open"):
        service.submit(io.BytesIO(b"not a video"), "notes.mp4", "test-camera", RECORDED_AT)
    assert service.store.waiting() == [] and sorted(p.name for p in service.folders.root.iterdir()) == ["jobs.sqlite", "service.lock"]


def test_an_upload_that_would_miss_the_deadline_is_refused(tmp_path, sample_video):
    service = start(tmp_path, deadline_minutes=0.6)  # 36 s: one job of 22 s fits, two do not.
    submit(service, sample_video)
    with pytest.raises(Refused, match="The queue is full: this video would be reviewed in about 1 min, after the deadline of 0.6 min") as refused:
        submit(service, sample_video)
    assert refused.value.status == 503 and len(service.store.waiting()) == 1


def test_a_job_runs_to_its_records_and_files(tmp_path, sample_video):
    service = start(tmp_path)
    job = submit(service, sample_video)
    with pytest.raises(NotReady, match="queued"):
        service.records(job.job_id)
    done = service.run_next()
    assert (done.state, done.stage, done.frames_done, done.frames_total, done.failure) == (DONE, "evidence", 240, 240, None)
    records = service.records(job.job_id)["events"]
    image = records[0]["evidence"]["frames"][0]["image"]
    assert len(records) == 2 and service.file(job.job_id, image).name.endswith("_before.jpg") and service.run_next() is None


@pytest.mark.parametrize("job_id, path", [("20261107-010000-abcdef", "records.json"), ("..", "records.json"), (None, "../jobs.sqlite"), (None, "nothing.jpg")])
def test_an_unknown_job_or_file_is_not_found(tmp_path, sample_video, job_id, path):
    service = start(tmp_path)
    job = submit(service, sample_video)
    with pytest.raises(NotFound):
        service.file(job_id or job.job_id, path)


def test_a_job_that_fails_keeps_its_reason_and_the_queue_goes_on(tmp_path, sample_video):
    service = start(tmp_path)
    broken, fine = submit(service, sample_video), submit(service, sample_video)
    (service.folders.folder(broken.job_id) / "video.mp4").unlink()
    first = service.run_next()
    assert (first.job_id, first.state) == (broken.job_id, FAILED) and first.failure.startswith("FileNotFoundError: Video not found")
    assert service.run_next().state == DONE and service.status(fine.job_id).state == DONE


def test_after_a_restart_the_job_that_was_running_has_failed_and_the_queue_is_kept(tmp_path, sample_video):
    service = start(tmp_path)
    running, waiting = submit(service, sample_video), submit(service, sample_video)
    service.store.start(running.job_id, "2026-11-07T01:00:00+00:00")
    service.close()
    restarted = start(tmp_path)
    assert restarted.interrupted == [running.job_id] and restarted.status(running.job_id).failure == INTERRUPTED
    assert restarted.status(waiting.job_id).state == QUEUED


def test_finished_jobs_older_than_the_retention_are_deleted_with_their_folders(tmp_path, sample_video):
    clock = Clock()
    service = start(tmp_path, clock)
    job = submit(service, sample_video)
    service.run_next()
    clock.now += timedelta(days=8)
    service.close()
    later = start(tmp_path, clock)
    with pytest.raises(NotFound):
        later.status(job.job_id)
    assert not later.folders.folder(job.job_id).exists()


def test_a_second_service_on_the_same_storage_is_refused(tmp_path):
    first = start(tmp_path)
    with pytest.raises(RuntimeError, match="Another service is using"):
        start(tmp_path)
    first.close()
    start(tmp_path).close()  # Once the first has stopped, the folder is free.
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_service.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.core'`

- [ ] **Step 3: Write the code**

`src/tvd/service/core.py`:

```python
"""The service's core, without HTTP: submit a job, see its state, get its records and files, list the cameras.

The REST layer only translates between HTTP and these calls (spec, Section 8). Jobs run one at a time,
in order of arrival: ``run_next`` runs the oldest queued one, and a worker calls it in a loop.
"""

from __future__ import annotations

import fcntl
import json
import re
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

from tvd.camera import load_rule_defaults
from tvd.records import parse_recorded_at
from tvd.service.cameras import CameraCatalog
from tvd.service.intake import Refused, estimate_seconds, inspect_video, save_upload
from tvd.service.jobs import DONE, RUNNING, Job, JobStore
from tvd.service.runner import Pipeline
from tvd.service.settings import ServiceSettings
from tvd.service.storage import JobFolders

SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")


class NotFound(Exception):
    """No such job, or no such file in it."""


class NotReady(Exception):
    """The job is not done, so it has no records yet. The message is its state."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_text(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


class ReviewService:
    """Everything is checked when it starts: the camera files, the settings, and the models, which load once.

    A job that was running when the service last stopped is marked failed, and finished jobs older than the
    retention are deleted with their folders. Only one service may use a storage folder: a second one would
    run the same queue.
    """

    def __init__(self, settings: ServiceSettings, clock: Callable[[], datetime] = utc_now, pipeline: Pipeline | None = None) -> None:
        self.settings = settings
        self.clock = clock
        self.cameras = CameraCatalog(settings.files.cameras, load_rule_defaults(settings.files.rules), settings.jobs.seconds_per_video_second)
        self.folders = JobFolders(settings.storage.folder)
        self._lock = (self.folders.root / "service.lock").open("w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise RuntimeError(f"Another service is using {self.folders.root}") from error
        self.store = JobStore(self.folders.database)
        self.pipeline = pipeline if pipeline is not None else Pipeline(settings.files)
        self._admission = threading.Lock()  # Two uploads at once must not both fit in the same room in the queue.
        self.interrupted = self.store.interrupt(as_text(clock()))
        self.remove_old()

    def submit(self, upload: BinaryIO, filename: str, camera_id: str, recorded_at: str) -> Job:
        """Save, check and queue an upload. ``Refused`` says why it was not queued; then nothing of it is kept."""
        try:
            parse_recorded_at(recorded_at)
        except ValueError as error:
            raise Refused(str(error)) from error
        camera = self.cameras.get(camera_id)
        if camera is None:
            raise Refused(f"Unknown camera {camera_id!r}. Known cameras: {', '.join(self.cameras.ids)}")
        now = self.clock()
        job_id = self.folders.new_id(now)
        suffix = Path(filename).suffix.lower()
        video = "video" + (suffix if SUFFIX.fullmatch(suffix) else ".mp4")  # The name sent is never used as a path.
        try:
            save_upload(upload, self.folders.folder(job_id) / video, self.settings.storage.max_upload_mb * 2**20)
            facts = inspect_video(self.folders.folder(job_id) / video, camera)
            estimate = estimate_seconds(facts, camera, self.settings.jobs)
            with self._admission:
                waiting = self.queue_seconds()
                deadline = self.settings.jobs.deadline_minutes * 60
                if waiting + estimate > deadline:
                    raise Refused(
                        f"The queue is full: this video would be reviewed in about {(waiting + estimate) / 60:.0f} min, "
                        f"after the deadline of {deadline / 60:g} min. Send it again later.", 503,
                    )
                job = Job(job_id, camera_id, recorded_at, video, round(facts.seconds, 3), round(estimate, 1), as_text(now))
                self.store.add(job)
        except BaseException:
            self.folders.remove(job_id)
            raise
        return job

    def queue_seconds(self) -> float:
        """How long the jobs now waiting should take, from their estimates: for the running one, what is left of it."""
        now = self.clock()
        total = 0.0
        for job in self.store.waiting():
            if job.state == RUNNING and job.started_at:
                total += max(0.0, job.estimate_seconds - (now - datetime.fromisoformat(job.started_at)).total_seconds())
            else:
                total += job.estimate_seconds
        return total

    def status(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise NotFound(f"No job {job_id}")
        return job

    def records(self, job_id: str) -> dict[str, Any]:
        """The job's ``records.json``, once it is done."""
        job = self.status(job_id)
        if job.state != DONE:
            raise NotReady(job.state)
        return json.loads((self.folders.folder(job_id) / "records.json").read_text())

    def file(self, job_id: str, path: str) -> Path:
        """One file of a job's folder, such as ``evidence/<event id>_entry.jpg``."""
        self.status(job_id)
        found = self.folders.file(job_id, path)
        if found is None:
            raise NotFound(f"No file {path} in job {job_id}")
        return found

    def describe_cameras(self) -> list[dict[str, Any]]:
        return self.cameras.describe()

    def run_next(self) -> Job | None:
        """Run the oldest queued job to its end, and return it; None if no job is queued. A failure is kept with its reason."""
        job = self.store.next_queued()
        if job is None:
            return None
        self.store.start(job.job_id, as_text(self.clock()))
        try:
            camera = self.cameras.get(job.camera_id)
            if camera is None:
                raise RuntimeError(f"Camera {job.camera_id} is no longer configured")

            def report(stage: str, done: int, total: int) -> None:
                self.store.progress(job.job_id, stage, done, total)

            self.pipeline.run(job, camera, self.folders.folder(job.job_id), report)
        except Exception as error:
            self.store.fail(job.job_id, f"{type(error).__name__}: {error}", as_text(self.clock()))
        else:
            self.store.finish(job.job_id, as_text(self.clock()))
        return self.store.get(job.job_id)

    def remove_old(self) -> list[str]:
        """Delete the finished jobs older than the retention, with their folders. Waiting jobs are never deleted."""
        cutoff = as_text(self.clock() - timedelta(days=self.settings.storage.keep_days))
        old = [job.job_id for job in self.store.finished_before(cutoff)]
        for job_id in old:
            self.folders.remove(job_id)
            self.store.delete(job_id)
        return old

    def close(self) -> None:
        self.pipeline.close()
        self._lock.close()  # Closing the file releases the lock.
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `323 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/service_data.py tests/tvd/test_tvd_service.py src/tvd/service/core.py
git commit -m "Submit, run and report jobs, without HTTP"
```

### Task 11: A worker for the queue

**Why:** Jobs run in the background, one at a time, because they share the GPU (spec, Section 5). The worker calls `run_next` in a loop, and waits a little when the queue is empty. Stopping it lets the running job end first. An unexpected error does not end the loop: a dead loop would leave every later job waiting, and nobody would know.

**Files:**
- Create: `tests/tvd/test_tvd_worker.py`
- Create: `src/tvd/service/worker.py`

**Interfaces:**
- Consumes: `ReviewService` (Task 10).
- Produces: `Worker(service, idle_seconds=2.0)` with `start()`, `stop(timeout=None)`, `running` and `errors`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_worker.py`:

```python
import time

from service_data import settings_for

from tvd.service.core import ReviewService
from tvd.service.jobs import DONE
from tvd.service.worker import Worker


def wait_for(condition, seconds: float = 60.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


def test_the_worker_runs_queued_jobs_in_order_until_stopped(tmp_path, sample_video):
    service = ReviewService(settings_for(tmp_path))
    with open(sample_video, "rb") as upload:
        first = service.submit(upload, "a.mp4", "test-camera", "2026-11-07T08:00:00+07:00")
    worker = Worker(service, idle_seconds=0.1)
    worker.start()
    try:
        assert wait_for(lambda: service.status(first.job_id).state == DONE)
        with open(sample_video, "rb") as upload:
            second = service.submit(upload, "b.mp4", "test-camera", "2026-11-07T08:05:00+07:00")
        assert wait_for(lambda: service.status(second.job_id).state == DONE)  # A job that arrives later is picked up too.
        assert service.status(first.job_id).finished_at <= service.status(second.job_id).started_at
    finally:
        worker.stop(timeout=30)
    assert not worker.running


def test_a_stopped_worker_takes_no_more_jobs(tmp_path, sample_video):
    service = ReviewService(settings_for(tmp_path))
    worker = Worker(service, idle_seconds=0.1)
    worker.start()
    worker.stop(timeout=30)
    with open(sample_video, "rb") as upload:
        job = service.submit(upload, "a.mp4", "test-camera", "2026-11-07T08:00:00+07:00")
    time.sleep(0.5)
    assert not worker.running and service.status(job.job_id).state == "queued"


class Flaky:
    """A service whose first call fails, as with a database locked for a moment."""

    def __init__(self) -> None:
        self.calls = 0

    def run_next(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("database is locked")
        return None

    def remove_old(self):
        return []


def test_the_worker_outlives_an_unexpected_error():
    service = Flaky()
    worker = Worker(service, idle_seconds=0.05)
    worker.start()
    try:
        assert wait_for(lambda: service.calls >= 3, seconds=5)
    finally:
        worker.stop(timeout=5)
    assert worker.errors == ["RuntimeError: database is locked"]
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_worker.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.worker'`

- [ ] **Step 3: Write the code**

`src/tvd/service/worker.py`:

```python
"""The thread that runs the queued jobs, one at a time: the GPU is shared, so jobs never run side by side."""

from __future__ import annotations

import threading

from tvd.service.core import ReviewService


class Worker:
    """Runs ``service.run_next`` until stopped, and waits ``idle_seconds`` whenever the queue is empty.

    After each job, finished jobs older than the retention are deleted. ``stop`` lets the running job end first.
    An unexpected error is kept in ``errors``, and the loop goes on: a dead loop would leave every later job waiting.
    """

    def __init__(self, service: ReviewService, idle_seconds: float = 2.0) -> None:
        self.service = service
        self.idle_seconds = idle_seconds
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="tvd-jobs", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.service.run_next()
                if job is not None:
                    self.service.remove_old()
            except Exception as error:  # A job's own failure is kept by run_next: this is anything else, such as a locked database.
                self.errors.append(f"{type(error).__name__}: {error}")
                job = None
            if job is None:
                self._stop.wait(self.idle_seconds)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `326 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_worker.py src/tvd/service/worker.py
git commit -m "Run the queued jobs in a thread of their own"
```

### Task 12: The REST API

**Why:** The five endpoints of the spec (Section 8), with FastAPI. Each one only calls the core: `Refused` becomes its status and reason, `NotFound` a 404, and records asked for too early a 409. An upload whose announced size is over the limit is refused before it is read. The worker starts and stops with the app.

**Files:**
- Create: `tests/tvd/test_tvd_api.py`
- Create: `src/tvd/service/api.py`

**Interfaces:**
- Consumes: `ReviewService`, `NotFound`, `NotReady` (Task 10); `Refused` (Task 6); `Worker` (Task 11).
- Produces: `create_app(service, run_worker=True) -> FastAPI` with `POST /jobs`, `GET /jobs/{job_id}`, `GET /jobs/{job_id}/records`, `GET /jobs/{job_id}/files/{path}` and `GET /cameras`.

- [ ] **Step 1: Write the failing test**

FastAPI's `TestClient` sends real HTTP requests to the app, in the test's process. Most tests run without the worker and call `run_next` themselves; the last one lets the worker do it.

`tests/tvd/test_tvd_api.py`:

```python
import time

import pytest
from fastapi.testclient import TestClient
from service_data import settings_for

from tvd.service.api import create_app
from tvd.service.core import ReviewService

FORM = {"camera_id": "test-camera", "recorded_at": "2026-11-07T08:00:00+07:00"}


@pytest.fixture
def service(tmp_path):
    return ReviewService(settings_for(tmp_path))


@pytest.fixture
def client(service):
    return TestClient(create_app(service, run_worker=False))


def upload(client, video, **form):
    with open(video, "rb") as data:
        return client.post("/jobs", files={"video": ("cam1.mp4", data, "video/mp4")}, data={**FORM, **form})


def test_the_cameras_are_listed(client):
    assert client.get("/cameras").json() == [{"camera_id": "test-camera", "frame_size": [640, 360], "motion": "fixed", "moving": False, "rules": ["red_light", "wrong_way"]}]


def test_an_upload_is_accepted_with_where_to_follow_it(client, sample_video):
    response = upload(client, sample_video)
    job = response.json()
    assert response.status_code == 202 and job["state"] == "queued" and job["status_url"] == f"/jobs/{job['job_id']}"
    assert client.get(job["status_url"]).json()["estimate_seconds"] == 22.0


@pytest.mark.parametrize("form, status, detail", [
    ({"recorded_at": "2026-11-07T08:00:00"}, 422, "needs a time zone"),
    ({"camera_id": "cam9"}, 422, "Unknown camera 'cam9'"),
])
def test_a_refused_upload_says_why(client, sample_video, form, status, detail):
    response = upload(client, sample_video, **form)
    assert response.status_code == status and detail in response.json()["detail"]


def test_an_upload_without_a_camera_is_refused_by_the_api_itself(client, sample_video):
    with open(sample_video, "rb") as data:
        response = client.post("/jobs", files={"video": ("cam1.mp4", data, "video/mp4")}, data={"recorded_at": FORM["recorded_at"]})
    assert response.status_code == 422 and response.json()["detail"][0]["loc"] == ["body", "camera_id"]


def test_an_upload_larger_than_the_limit_is_refused_before_it_is_read(tmp_path):
    settings = settings_for(tmp_path)
    small = settings.model_copy(update={"storage": settings.storage.model_copy(update={"max_upload_mb": 1})})
    client = TestClient(create_app(ReviewService(small), run_worker=False))
    response = client.post("/jobs", files={"video": ("big.mp4", b"x" * (2 * 2**20), "video/mp4")}, data=FORM)
    assert response.status_code == 413 and response.json()["detail"] == "The upload is larger than 1 MB"


def test_records_come_when_the_job_is_done_and_its_files_can_be_fetched(client, service, sample_video):
    job_id = upload(client, sample_video).json()["job_id"]
    early = client.get(f"/jobs/{job_id}/records")
    assert early.status_code == 409 and early.json()["detail"] == f"Job {job_id} is queued: its records come when it is done"
    service.run_next()
    records = client.get(f"/jobs/{job_id}/records").json()["events"]
    image = client.get(f"/jobs/{job_id}/files/{records[1]['evidence']['frames'][1]['image']}")
    assert len(records) == 2 and image.status_code == 200 and image.headers["content-type"] == "image/jpeg" and image.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize("path", ["/jobs/20261107-010000-abcdef", "/jobs/../cameras/records", "/jobs/{job}/files/..%2Fjobs.sqlite", "/jobs/{job}/files/nothing.jpg"])
def test_an_unknown_job_or_file_is_not_found(client, sample_video, path):
    job_id = upload(client, sample_video).json()["job_id"]
    assert client.get(path.replace("{job}", job_id)).status_code == 404


def test_with_its_worker_the_app_runs_each_upload_to_its_records(service, sample_video):
    with TestClient(create_app(service)) as client:
        job_id = upload(client, sample_video).json()["job_id"]
        end = time.monotonic() + 60
        while client.get(f"/jobs/{job_id}").json()["state"] in ("queued", "running") and time.monotonic() < end:
            time.sleep(0.2)
        assert client.get(f"/jobs/{job_id}").json()["state"] == "done" and len(client.get(f"/jobs/{job_id}/records").json()["events"]) == 2
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_api.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.api'`

- [ ] **Step 3: Write the code**

`src/tvd/service/api.py`:

```python
"""The REST layer (spec, Section 8). It only translates between HTTP and the service's core.

| Method | Path | Answer |
|---|---|---|
| POST | /jobs | 202 and the job; 413, 422 or 503 with the reason, and nothing is queued |
| GET | /jobs/{job_id} | the job's state, stage, progress and failure |
| GET | /jobs/{job_id}/records | records.json once the job is done; 409 before |
| GET | /jobs/{job_id}/files/{path} | one file of the job's folder |
| GET | /cameras | the configured cameras and their rules |
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from tvd.service.core import NotFound, NotReady, ReviewService
from tvd.service.intake import Refused
from tvd.service.worker import Worker


def create_app(service: ReviewService, run_worker: bool = True) -> FastAPI:
    """The API of one service. With ``run_worker``, its jobs run in the background while the app runs."""
    worker = Worker(service) if run_worker else None
    limit = service.settings.storage.max_upload_mb * 2**20

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()  # The running job ends first.
            service.close()

    app = FastAPI(title="Traffic violation review", version="1", lifespan=lifespan)

    @app.middleware("http")
    async def refuse_large_uploads(request: Request, call_next):
        """A body announced larger than the limit is refused before it is read. Without a length, the upload's own check refuses it."""
        length = request.headers.get("content-length", "")
        if request.method == "POST" and request.url.path == "/jobs" and length.isdigit() and int(length) > limit:
            return JSONResponse(status_code=413, content={"detail": f"The upload is larger than {limit / 2**20:.0f} MB"})
        return await call_next(request)

    @app.exception_handler(Refused)
    async def refused(request: Request, error: Refused) -> JSONResponse:
        return JSONResponse(status_code=error.status, content={"detail": error.reason})

    @app.exception_handler(NotFound)
    async def not_found(request: Request, error: NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.post("/jobs", status_code=202)
    def submit(video: UploadFile = File(...), camera_id: str = Form(...), recorded_at: str = Form(...)) -> dict[str, Any]:
        job = service.submit(video.file, video.filename or "", camera_id, recorded_at)
        return {**job.as_json(), "status_url": f"/jobs/{job.job_id}", "records_url": f"/jobs/{job.job_id}/records"}

    @app.get("/jobs/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        return service.status(job_id).as_json()

    @app.get("/jobs/{job_id}/records")
    def records(job_id: str) -> dict[str, Any]:
        try:
            return service.records(job_id)
        except NotReady as state:
            raise HTTPException(status_code=409, detail=f"Job {job_id} is {state}: its records come when it is done") from None

    @app.get("/jobs/{job_id}/files/{path:path}")
    def file(job_id: str, path: str) -> FileResponse:
        return FileResponse(service.file(job_id, path))

    @app.get("/cameras")
    def cameras() -> list[dict[str, Any]]:
        return service.describe_cameras()

    return app
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `338 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_api.py src/tvd/service/api.py
git commit -m "Serve the jobs over REST"
```

### Task 13: The serve command

**Why:** The last piece: `python -m tvd.cli serve` checks every setting and camera file, loads the models, and then listens. The REST layer's packages are imported only here, so the other commands do not need them. The steps after the tests run the real service on the server, and send it a video as your backend will.

**Files:**
- Change: `tests/tvd/test_tvd_cli.py`
- Change: `src/tvd/cli.py`

**Interfaces:**
- Consumes: Everything above.
- Produces: `python -m tvd.cli serve [--config configs/service.yaml]`; `run_serve(config_path)`.

- [ ] **Step 1: Write the failing test**

The tests replace `uvicorn.run`, so that nothing listens: the first checks what would be served and where, and the second that a broken camera file stops the service before it listens.

In `tests/tvd/test_tvd_cli.py`:

After:

```python
from notebook_data import NOTEBOOK_08_RLR, RED_LIGHT_SCENE, RLR_VIDEO, cut_losslessly, load_json
```

add:

```python
from service_data import service_files
```

Add to the end of the file:

```python
def test_serve_checks_everything_then_listens_where_the_settings_say(tmp_path, monkeypatch):
    import uvicorn
    from fastapi import FastAPI

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: started.update(app=app, host=host, port=port))
    assert main(["serve", "--config", str(service_files(tmp_path))]) == 0
    assert (started["host"], started["port"]) == ("127.0.0.1", 8040) and isinstance(started["app"], FastAPI)


def test_serve_refuses_a_broken_camera_file_before_it_listens(tmp_path, capsys, monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: pytest.fail("the service must not start"))
    config = service_files(tmp_path)
    camera = tmp_path / "cameras" / "test-camera.yaml"
    camera.write_text(camera.read_text().replace("pass_margin", "pass_margn"))
    assert main(["serve", "--config", str(config)]) == 2
    assert "test-camera.yaml: rules.red_light:" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_cli.py -q
```

Expected: `2 failed, 10 passed`: both stop with `SystemExit: 2`, after `tvd: error: argument command: invalid choice: 'serve' (choose from 'track', 'check', 'review')`.

- [ ] **Step 3: Write the code**

In `src/tvd/cli.py`:

After:

```python
  each candidate's evidence and record, for a person to review.
```

add:

```python
- ``python -m tvd.cli serve [--config configs/service.yaml]``: the review service, with its REST API.
```

After:

```python
EVIDENCE = Path("configs/tvd/evidence.yaml")
```

add:

```python
SERVICE = Path("configs/service.yaml")
```

Before:

```python
def main(argv: list[str] | None = None) -> int:
```

add:

```python
def run_serve(config_path: Path) -> None:
    """Every setting and camera file is checked, and the models load, before the service listens. Ctrl+C stops it.

    The REST layer's packages are imported here only: the other commands do not need them.
    """
    import uvicorn

    from tvd.service.api import create_app
    from tvd.service.core import ReviewService
    from tvd.service.settings import load_service_settings

    settings = load_service_settings(config_path)
    service = ReviewService(settings)
    for job_id in service.interrupted:
        print(f"job {job_id} was running when the service stopped: it is marked failed", file=sys.stderr)
    uvicorn.run(create_app(service), host=settings.server.host, port=settings.server.port)
```

After:

```python
    review.add_argument("--evidence", type=Path, default=EVIDENCE)
```

add:

```python
    serve = commands.add_parser("serve", help="Run the review service and its REST API")
    serve.add_argument("--config", type=Path, default=SERVICE)
```

Replace:

```python
            print(check_summary(run_check(args.video, args.camera, args.output, args.perception, args.rules)))
        else:
```

with:

```python
            print(check_summary(run_check(args.video, args.camera, args.output, args.perception, args.rules)))
        elif args.command == "review":
```

After:

```python
            result = run_review(args.video, args.camera, args.recorded_at, args.output, args.perception, args.rules, args.plates, args.evidence)
            print(review_summary(result))
```

add:

```python
        else:
            run_serve(args.config)
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `340 passed`.

- [ ] **Step 5: Start the service**

This runs the real service on the server, with `configs/service.yaml`: it listens on port 8040 of the server only, and keeps its jobs in `outputs/service/`. Start it in the background, and keep its log:

The braces matter: `scripts/server.sh run` puts `cd ... &&` before the command, and without them `&` would send that whole line to the background, which keeps the ssh session, and your terminal, waiting.

```bash
scripts/server.sh run '{ setsid nohup python -m tvd.cli serve > /tmp/tvd-serve.log 2>&1 < /dev/null & }'
scripts/server.sh run 'sleep 20; curl -s localhost:8040/cameras'
```

Expected, once the models have loaded: the three cameras, beginning with `[{"camera_id":"rlr_occasions-C","frame_size":[1280,720],"motion":"fixed","moving":false,"rules":["red_light"]}`.

- [ ] **Step 6: Upload a video and follow it**

Send a video the way your backend will, with the camera and the time of its first frame. First with the wrong camera, then the right one:

```bash
scripts/server.sh run 'curl -s -F video=@data/traffic_video_modified.mp4 -F camera_id=video-test -F recorded_at=2026-11-07T19:30:00+07:00 localhost:8040/jobs'
scripts/server.sh run 'curl -s -F video=@data/traffic_video_modified.mp4 -F camera_id=traffic_video_modified -F recorded_at=2026-11-07T19:30:00+07:00 localhost:8040/jobs'
```

Expected: first `{"detail":"Camera video-test has 1920x1080 frames, but video.mp4 is 1920x1012"}` (status 422: nothing is queued), then the job, with `"state":"queued"`, an `estimate_seconds` of about 190, and its `job_id`, such as `20261107-123456-0a1b2c`. Then follow it, with that id:

```bash
scripts/server.sh run 'curl -s localhost:8040/jobs/<job id>'
```

Expected: `"state":"running"` with `"stage":"tracking"` and `frames_done` growing to 715, then `"stage":"camera motion"`, then `"evidence"`, and after about 2 minutes `"state":"done"`.

- [ ] **Step 7: Get the results, and stop the service**

Get the records, a clip and an image over HTTP, and stop the service:

```bash
scripts/server.sh run 'curl -s localhost:8040/jobs/<job id>/records | python -c "import json, sys; r = json.load(sys.stdin)[\"events\"]; print(len(r), [e[\"license_plate\"][\"status\"] for e in r])"'
scripts/server.sh run 'curl -s localhost:8040/jobs/<job id>/files/clips/evt-<job id>-red_light-f00317-t23.mp4 | cmp - outputs/08_end_to_end_baseline/traffic_video_modified/violations/traffic_video_modified_violation_clips/evt-traffic_video_modified-main-red_light-f00317-t23.mp4 && echo "clip: same"'
scripts/server.sh run 'curl -s -o /dev/null -w "%{http_code} %{content_type}\n" localhost:8040/jobs/<job id>/files/evidence/evt-<job id>-red_light-f00317-t23_entry.jpg'
scripts/server.sh run 'pkill -INT -f "python -m tvd.cli serve"'
```

Expected: `2 ['read', 'read']` (the plate texts are not printed), `clip: same`, and `200 image/jpeg`. The job's folder, `outputs/service/<job id>/`, holds the video and its review, and is deleted 7 days after the job ends. Started again, the service keeps reporting this job.

- [ ] **Step 8: Commit**

```bash
git add tests/tvd/test_tvd_cli.py src/tvd/cli.py
git commit -m "Add the serve command"
```

---

## When Part 4 is done

- [ ] `scripts/server.sh test tests/tvd -q` gives `340 passed`, and `scripts/server.sh test -q` still passes the notebooks' tests too.
- [ ] The service, started on the server, takes an upload, reports its progress, and gives its records, clips and images over HTTP (Task 13, Steps 5 to 7).
- [ ] A wrong camera, a time without a time zone, or a file that is not a video is refused with a reason, and leaves nothing in `outputs/service/`.
- [ ] Stopped in the middle of a job and started again, the service reports that job as failed, and runs the jobs still queued.
- [ ] You can say which file to change for: a longer deadline, another port, a new camera, a slower server.

**Exercises, if you want them** (not checked while writing the plan):

- **A message queue instead of REST**: a second front door that reads uploads from a queue and calls `ReviewService.submit`. The core does not change: this is why it has no HTTP in it.
- **A better estimate**: after each job, compare `estimate_seconds` with the time it really took, and keep the ratio in `summary.json`. A few days of jobs tell you whether the measured speeds of `service.yaml` still hold.

**Next:** Part 5, deployment and MLOps, puts the service in a Docker image, runs it on ai_server_2 as a service that restarts by itself, and adds logging, a health check and the versions of the models to every record. Its plan is written once Part 4 is done.
