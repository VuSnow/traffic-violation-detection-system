# Service Part 5: Deployment and MLOps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the review service on ai_server_2 as a Docker container that Docker starts again by itself, with a JSON log, a health check, and the versions of its models and software in every record.

**Architecture:** Each model describes itself (weights and their hash, device, ONNX providers), and every record keeps that description with the libraries' versions and the code's commit. The service logs each job's events as JSON lines, to a rotated file and to standard output, and `GET /health` says whether it is ok, degraded or failing. A Docker image holds the exact versions and models the tests passed with, and `scripts/deploy.sh` builds it and runs it on the GPU, on port 8040 of the server only.

**Tech Stack:** Docker 27 with the NVIDIA container runtime (already on ai_server_2), `python:3.11.18-slim-bookworm`, Python's `logging`, and Parts 1 to 4.

**Spec:** `docs/superpowers/specs/2026-11-07-review-service-design.md` (Sections 4, 7, 10 and 12: "Docker, running on ai_server_2, logging, health checks, model versions").

**Written from the code:** Part 5 was built first, at your request, and this plan was written from its commits on `main` (`5132e49` to `2b7d514`), so that you can build it again. Every code block below is that code; each failure and count is what the tests gave on ai_server_2 while it was built. Two mistakes made on the way are not repeated here; they are explained at the end, under **What building it taught**.

## The five parts

| Part | Builds | Plan |
|---|---|---|
| 1. Foundations and perception | Data types, registry, video reader, detector, trackers, settings, `track` command | Done: `2026-11-07-service-part-1-foundations-and-perception.md` |
| 2. Scene and rules | Camera files, rule settings, camera motion, signal reading, the two rules, `check` command | Done: `2026-11-07-service-part-2-scene-and-rules.md` |
| 3. Plates and evidence | Clips, plate models in a process of their own, voting, evidence images, records, `review` command | Done: `2026-11-07-service-part-3-plates-and-evidence.md` |
| 4. The service | Settings, camera catalog, upload checks, job queue, job folders, worker, REST API, `serve` command | Done: `2026-11-07-service-part-4-the-service.md` |
| **5. Deployment and MLOps** | Versions in records, JSON log, health check, Docker image, deployment script | This document |

## What runs on the server

```text
laptop ── ssh -L 8040 ──┐
                        ▼
ai_server_2   127.0.0.1:8040 ──> container tvd-service (image tvd-service:<commit>, --gpus all, --restart unless-stopped)
                                    python -m tvd.cli serve --config configs/service.docker.yaml
                                      GET /health ──> ok | degraded | failing (503): Docker's health check calls it
                                      jobs ──> records with "models": weights and their hash, device, providers, versions, commit
                                      log  ──> JSON lines on standard output (docker logs) and in /data/service/logs
              outputs/service/  <── mounted at /data/service: the jobs, their database, the log
```

## Global Constraints

- **ai_server_2 is shared.** Nothing is installed or changed on the server itself: Docker and the NVIDIA runtime are already there, and the user `ubuntu` is in the `docker` group. The service's port is published on `127.0.0.1` only.
- **The image runs what the tests passed with**: the server's Python (3.11.18), every package at the version of the `tvd` environment (`docker/constraints.txt`), and the model files the tests used. It downloads nothing when it is built or when it runs.
- **The plate models stay out of git**: their licence is not stated. They go from the server's cache to `docker/models/`, which git ignores, and into the image.
- **Nothing tunable is in code**: the log's folder, level and sizes, and the least free space, are in `configs/service.yaml`; the container's own differences are in `configs/service.docker.yaml`.
- **No log line holds a plate number**, as no output of Parts 3 and 4 does.
- Never let pip change OpenCV (5.0.0.93), NumPy (2.4.6) or pydantic (2.14.3).
- Tests that need the GPU stack or the plate models skip themselves elsewhere. The counts below are from the GPU server.

## Review Focus

- **The shared GPU filled by another user's job**: the plate models then start on the CPU, slowly and with slightly different reads. The health check must say `degraded`, not `ok`: `test_a_service_that_does_jobs_less_well_is_degraded` (Task 5).
- **A worker that died, or a full disk**: no job will be done, and Docker must be told so: `test_a_service_that_cannot_do_jobs_is_failing` (Task 5), `test_health_is_failing_without_a_worker` (Task 5).
- **A record or a plate reaching the log**: `test_no_log_line_holds_a_record_or_a_plate` (Task 4).
- **An image with other versions than the tested ones**: `test_the_pins_are_the_versions_installed_where_the_tests_run`, `test_the_image_pins_the_versions_the_code_was_tested_with` (Task 6).
- **A crash of the service**: Docker must start it again, and the service must still know its jobs: Task 8, the last check.

## How you will work

As in Parts 1 to 4: edit on your laptop, run on ai_server_2 with `scripts/server.sh`. From Task 6 on, `scripts/server.sh sync` also sends `docker/`, `.dockerignore` and `.gitignore`, which the image needs. Docker commands run on the server, through `scripts/server.sh run scripts/deploy.sh ...`.

## File structure

```text
src/tvd/versions.py              Task 1/2  PACKAGES, package_versions, code_version, models_used
src/tvd/detection/base.py, yolo.py Task 1  describe(): weights, their version, device         (changed)
src/tvd/plates/base.py, onnx.py  Task 1    describe(): models and ONNX providers               (changed)
src/tvd/records.py               Task 2    RecordContext.models, a record's "models"           (changed)
src/tvd/service/logs.py          Task 3/4  JsonFormatter, log_event, setup_logging
src/tvd/service/settings.py      Task 3/5  LoggingSettings, StorageSettings.min_free_gb        (changed)
src/tvd/service/core.py          Task 4/5  job events; health()                                (changed)
src/tvd/service/runner.py        Task 2/4/5 models in records; plate model events; models()    (changed)
src/tvd/service/worker.py        Task 4    worker_error                                        (changed)
src/tvd/service/api.py           Task 5    GET /health                                         (changed)
src/tvd/cli.py                   Task 2/5  models in the review's records; serve logs           (changed)
configs/service.yaml             Task 3/5  logging; storage.min_free_gb                         (changed)
configs/service.docker.yaml      Task 6    the same, for the container
docker/Dockerfile                Task 6    the image
docker/constraints.txt           Task 6    every version of the tested environment
docker/copy_plate_models.py      Task 6    the plate models, from the server's cache to docker/models/
docker/check_plate_models.py     Task 6    load them while the image is built
.dockerignore, .gitignore        Task 6    what Docker and git may see                          (.gitignore changed)
scripts/server.sh                Task 6    sync sends docker/, .dockerignore, .gitignore        (changed)
scripts/deploy.sh                Task 7    build, start, stop, status, logs, health
docs/event_schema.md             Task 2    a record's "models"                                  (changed)
```

## SOLID and patterns in this part

| Principle or pattern | Where |
|---|---|
| **S**ingle responsibility | `versions.py` names what produced a result, `logs.py` writes lines, `health()` judges, `deploy.sh` deploys |
| **O**pen/closed | `describe()` has a default in each base class: a new detector or plate model works at once, and says more by overriding it |
| **L**iskov substitution | Every detector and set of plate models can describe itself; the fakes of the tests do it with the default |
| **D**ependency inversion | The service writes events through `log_event`; where they go is set once, by `setup_logging`, when it starts |
| Health check | `GET /health` and Docker's `HEALTHCHECK` |
| Immutable image | One image per commit, tagged with it, holding its versions and models |

---

### Task 0: Branch, and what the server offers

**Why:** Part 5 runs the service in Docker on ai_server_2, with the GPU. The server is shared: this part must not install or change anything on it. This task checks that what it needs is there.

**Files:** none.

- [ ] **Step 1: Start the branch from `main`, where Part 4 is merged**

```bash
git checkout main
git checkout -b service/part-5
```

- [ ] **Step 2: Check that Part 4 passes on the server**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `340 passed`.

- [ ] **Step 3: Check that you can use Docker and the GPU in containers on the server, without sudo**

```bash
scripts/server.sh run 'id -nG | tr " " "\n" | grep -x docker; docker info 2>/dev/null | grep -i runtimes; df -h / | tail -1'
```

Expected: `docker`, then a `Runtimes:` line that lists `nvidia`, then the disk: the image needs about 8 GB. If `docker` or `nvidia` is missing, stop and ask: installing them changes the shared server.

### Task 1: Versions of the code, the libraries and the models

**Why:** A record must say what produced it (spec, Section 4), and a result can change when a model or a library changes. Each detector and each set of plate models now describes itself: the YOLO detector names its weights, their version (the hash of the file) and its device; the ONNX models name their models and the providers they really run on, which say whether the GPU is used. The code's version is the commit the Docker image was built from (Task 6).

**Files:**
- Create: `tests/tvd/test_tvd_versions.py`
- Change: `tests/tvd/test_tvd_detection.py`
- Change: `tests/tvd/test_tvd_plates.py`
- Change: `tests/tvd/test_tvd_onnx_plates.py`
- Create: `src/tvd/versions.py`
- Change: `src/tvd/detection/base.py`
- Change: `src/tvd/detection/yolo.py`
- Change: `src/tvd/plates/base.py`
- Change: `src/tvd/plates/onnx.py`

**Interfaces:**
- Consumes: `file_version` (Part 3); `BaseDetector`, `YoloDetector` (Part 1); `BasePlateModels`, `OnnxPlateModels` (Parts 3 and 4).
- Produces: `PACKAGES`, `package_versions(names=PACKAGES) -> dict[str, str | None]` and `code_version() -> str | None` (from `TVD_COMMIT`) in `tvd/versions.py`; `BaseDetector.describe()` and `BasePlateModels.describe() -> dict`, which give `{"type": <class name>}` by default; `YoloDetector.describe()` with `weights`, `weights_version` and `device`; `OnnxPlateModels.describe()` with `detector_model`, `ocr_model` and `providers`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_versions.py`:

```python
from tvd.versions import PACKAGES, code_version, package_versions


def test_the_versions_of_the_libraries_that_shape_results():
    versions = package_versions()
    assert list(versions) == list(PACKAGES) and versions["traffic-violation-detection"] == "0.1.0" and versions["numpy"] == "2.4.6"


def test_a_package_that_is_not_installed_has_no_version():
    assert package_versions(("no-such-package",)) == {"no-such-package": None}


def test_the_code_version_comes_from_the_build(monkeypatch):
    monkeypatch.setenv("TVD_COMMIT", "6b1a1ba")
    assert code_version() == "6b1a1ba"
    monkeypatch.delenv("TVD_COMMIT")
    assert code_version() is None
```

In `tests/tvd/test_tvd_detection.py`:

Before:

```python
from tvd.detection import DETECTORS, BaseDetector
```

add:

```python
from tvd.config import file_version
```

Add to the end of the file:

```python
def test_a_detector_describes_itself_by_its_class_by_default():
    assert DETECTORS.create("test-moving-car").describe() == {"type": "MovingCarDetector"}


@needs_yolo
def test_yolo_names_its_weights_their_version_and_its_device(monkeypatch):
    import torch

    monkeypatch.chdir(ROOT)
    assert DETECTORS.create("yolo", weights="yolo26m.pt").describe() == {
        "type": "yolo", "weights": "yolo26m.pt", "weights_version": file_version(ROOT / "weights" / "yolo26m.pt"),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
```

In `tests/tvd/test_tvd_plates.py`:

After:

```python
        assert models.recognize(IMAGE).characters == "FAKE9999_"
    assert models.closed
```

add:

```python


def test_plate_models_describe_themselves_by_their_class_by_default():
    assert FakePlateModels().describe() == {"type": "FakePlateModels"}
```

In `tests/tvd/test_tvd_onnx_plates.py`:

After:

```python
def test_the_models_run_in_another_process(models):
    assert models.pid != os.getpid()
```

add:

```python


def test_the_models_say_what_they_are_and_where_they_run(models):
    assert models.describe() == {
        "type": "onnx_worker", "detector_model": "yolo-v9-t-640-license-plate-end2end", "ocr_model": "cct-s-v2-global-model", "providers": models.providers,
    }
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_versions.py tests/tvd/test_tvd_detection.py tests/tvd/test_tvd_plates.py tests/tvd/test_tvd_onnx_plates.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.versions'`

- [ ] **Step 3: Write the code**

`src/tvd/versions.py`:

```python
"""What produced a result: the code, the libraries that shape it, and the model files. Records and the health check keep them."""

from __future__ import annotations

import os
from importlib import metadata

PACKAGES = (
    "traffic-violation-detection", "torch", "ultralytics", "boxmot", "onnxruntime-gpu", "fast-plate-ocr", "open-image-models", "opencv-python", "numpy",
)


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """The installed version of each package, or None if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_version() -> str | None:
    """The commit the code was built from, which the Docker image keeps in ``TVD_COMMIT``. None outside an image."""
    return os.environ.get("TVD_COMMIT") or None
```

In `src/tvd/detection/base.py`:

After:

```python
from abc import ABC, abstractmethod
```

add:

```python
from typing import Any
```

Before:

```python

DETECTORS: Registry[BaseDetector] = Registry("detector")
```

add:

```python
    def describe(self) -> dict[str, Any]:
        """What this detector is, for records and the health check. A model's detector names its weights and device."""
        return {"type": type(self).__name__}
```

In `src/tvd/detection/yolo.py`:

After:

```python
"""Ultralytics YOLO models, behind ``BaseDetector``."""

from pathlib import Path
```

add:

```python
from typing import Any
```

Before:

```python
from tvd.detection.base import DETECTORS, BaseDetector
```

add:

```python
from tvd.config import file_version
```

Replace:

```python
        self._model = YOLO(str(Path(weights_dir) / weights))
```

with:

```python
        self._weights = Path(weights_dir) / weights
        self._model = YOLO(str(self._weights))
```

After:

```python
        self._class_ids = None if classes is None else self._ids_of(classes, weights)
```

add:

```python

    def describe(self) -> dict[str, Any]:
        import torch

        # Ultralytics predicts on the first GPU when there is one.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return {"type": "yolo", "weights": self._weights.name, "weights_version": file_version(self._weights), "device": device}
```

In `src/tvd/plates/base.py`:

After:

```python
from dataclasses import dataclass
```

add:

```python
from typing import Any
```

Before:

```python
    def __enter__(self) -> BasePlateModels:
```

add:

```python
    def describe(self) -> dict[str, Any]:
        """What these models are, for records and the health check. Real models name themselves and where they run."""
        return {"type": type(self).__name__}
```

In `src/tvd/plates/onnx.py`:

After:

```python
        options = {"detector_model": detector_model, "ocr_model": ocr_model, "cpu_threads": cpu_threads}
```

add:

```python
        self.detector_model = detector_model
        self.ocr_model = ocr_model
```

Add to the end of the file:

```python
    def describe(self) -> dict[str, Any]:
        """``providers`` begins with ``CUDAExecutionProvider`` when the models run on the GPU, and with the CPU's when the GPU was full."""
        return {"type": "onnx_worker", "detector_model": self.detector_model, "ocr_model": self.ocr_model, "providers": self.providers}
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `347 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_versions.py tests/tvd/test_tvd_detection.py tests/tvd/test_tvd_plates.py tests/tvd/test_tvd_onnx_plates.py src/tvd/versions.py src/tvd/detection/base.py src/tvd/detection/yolo.py src/tvd/plates/base.py src/tvd/plates/onnx.py
git commit -m "Name the versions of the code, the libraries and the models"
```

### Task 2: The models behind each record

**Why:** Every record now carries `models`: the detector, the tracker, the plate models (or why there were none), the libraries' versions and the code's commit. Two records that differ can then be traced to what differed. `RecordContext` gets them once per video, so the review command (Part 3) and the service (Part 4) fill them the same way.

**Files:**
- Change: `tests/tvd/test_tvd_versions.py`
- Change: `tests/tvd/test_tvd_records.py`
- Change: `tests/tvd/test_tvd_review.py`
- Change: `tests/tvd/test_tvd_runner.py`
- Change: `tests/tvd/test_tvd_cli.py`
- Change: `src/tvd/versions.py`
- Change: `src/tvd/records.py`
- Change: `src/tvd/cli.py`
- Change: `src/tvd/service/runner.py`
- Change: `docs/event_schema.md`

**Interfaces:**
- Consumes: `describe()`, `package_versions`, `code_version` (Task 1); `RecordContext`, `build_record`, `validate_record` (Part 3).
- Produces: `models_used(detector, tracker, plate_models, plates_unavailable) -> dict` in `tvd/versions.py`; `RecordContext.models`, a new last field; a record's `models`, which `validate_record` now requires.

- [ ] **Step 1: Write the failing test**

In `tests/tvd/test_tvd_versions.py`:

Replace:

```python
from tvd.versions import PACKAGES, code_version, package_versions
```

with:

```python
from tvd.config import ComponentConfig
from tvd.detection import DETECTORS
from tvd.versions import PACKAGES, code_version, models_used, package_versions
```

Add to the end of the file:

```python
def test_the_models_used_name_each_model_and_the_software(monkeypatch):
    monkeypatch.setenv("TVD_COMMIT", "6b1a1ba")
    used = models_used(DETECTORS.create("test-moving-car"), ComponentConfig(type="ocsort", options={"use_byte": True}), None, "the plate models did not start: no GPU")
    assert used["detector"] == {"type": "MovingCarDetector"} and used["tracker"] == {"type": "ocsort", "use_byte": True}
    assert used["plates"] == {"unavailable": "the plate models did not start: no GPU"} and used["code"] == "6b1a1ba" and list(used["software"]) == list(PACKAGES)
```

In `tests/tvd/test_tvd_records.py`:

After:

```python
SETTINGS = {"perception": {"tracker": "ocsort"}, "rules": RULES, "plates": {"version": 1}, "evidence": {"version": 1}}
```

add:

```python
MODELS = {"detector": {"type": "yolo", "weights": "yolo26m.pt"}, "tracker": {"type": "ocsort"}, "plates": {"type": "onnx_worker"}, "software": {"numpy": "2.4.6"}, "code": None}
```

Replace:

```python
    return RecordContext(video_id, fps, 1920, 1080, "cam1", RECORDED_AT, f"{video_id}/ocsort", SETTINGS, {"camera": "sha256:0123456789ab"})
```

with:

```python
    return RecordContext(video_id, fps, 1920, 1080, "cam1", RECORDED_AT, f"{video_id}/ocsort", SETTINGS, {"camera": "sha256:0123456789ab"}, MODELS)
```

After:

```python
    assert r["rule"] == {"name": "red_light", "version": 2} and r["settings"]["rule"] == RULES["red_light"] and r["config_files"] == {"camera": "sha256:0123456789ab"}
```

add:

```python
    assert r["models"] == MODELS
```

After:

```python
    (lambda r: r.pop("camera_id"), "camera_id is missing"),
```

add:

```python
    (lambda r: r.pop("models"), "models is missing"),
```

In `tests/tvd/test_tvd_review.py`:

Replace:

```python
                                "traffic_sample/test-same-id", settings, {"camera": "sha256:000000000000"})
```

with:

```python
                                "traffic_sample/test-same-id", settings, {"camera": "sha256:000000000000"}, {"detector": {"type": "test"}})
```

In `tests/tvd/test_tvd_runner.py`:

After:

```python
    assert set(records[0]["config_files"]) == {"perception", "rules", "plates", "evidence", "camera"} and (folder / "video.mp4").is_file()
```

add:

```python
    models = records[0]["models"]
    assert (models["detector"], models["tracker"], models["plates"]) == ({"type": "MovingCarDetector"}, {"type": "test-same-id"}, {"type": "FakePlateModels"})
    assert models["software"]["numpy"] == "2.4.6" and "code" in models
```

In `tests/tvd/test_tvd_cli.py`:

After:

```python
    assert set(record["config_files"]) == {"perception", "camera", "rules", "plates", "evidence"} and record["settings"]["rule"]["pass_margin"] == 0.2
```

add:

```python
    assert record["models"]["tracker"] == {"type": "test-same-id"} and record["models"]["plates"] == {"type": "FakePlateModels"}
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_versions.py tests/tvd/test_tvd_records.py tests/tvd/test_tvd_review.py tests/tvd/test_tvd_runner.py tests/tvd/test_tvd_cli.py -q
```

Expected: `E   ImportError: cannot import name 'models_used' from 'tvd.versions'`

- [ ] **Step 3: Write the code**

In `src/tvd/versions.py`:

After:

```python
import os
from importlib import metadata
```

add:

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tvd.config import ComponentConfig
    from tvd.detection import BaseDetector
    from tvd.plates import BasePlateModels
```

Add to the end of the file:

```python
def models_used(detector: BaseDetector, tracker: ComponentConfig, plate_models: BasePlateModels | None, plates_unavailable: str) -> dict[str, Any]:
    """Every model and library behind a record: changing any of them may change what is found."""
    return {
        "detector": detector.describe(),
        "tracker": {"type": tracker.type, **tracker.options},
        "plates": plate_models.describe() if plate_models is not None else {"unavailable": plates_unavailable},
        "software": package_versions(),
        "code": code_version(),
    }
```

In `src/tvd/records.py`:

Replace:

```python
    "settings": dict, "config_files": dict, "review": dict,
```

with:

```python
    "settings": dict, "config_files": dict, "models": dict, "review": dict,
```

Replace:

```python
    and ``config_files`` the version of each settings file that gave them.
```

with:

```python
    ``config_files`` the version of each settings file that gave them, and ``models`` the models and software (``models_used``).
```

After:

```python
    settings: dict[str, Any]
    config_files: dict[str, str]
```

add:

```python
    models: dict[str, Any]
```

After:

```python
        "config_files": context.config_files,
```

add:

```python
        "models": context.models,
```

In `src/tvd/cli.py`:

After:

```python
from tvd.tracking.history import TrackHistory
```

add:

```python
from tvd.versions import models_used
```

After:

```python
                "plates": plates.model_dump(), "evidence": evidence.model_dump(),
            }
```

add:

```python
            unavailable = unavailable or "no plate models"
            models = models_used(detector, perception.tracker, plate_models, unavailable)
```

Replace:

```python
                video.stem, reader.fps, reader.width, reader.height, camera.camera_id, moment, f"{video.stem}/{perception.tracker.type}", settings, config_files,
```

with:

```python
                video.stem, reader.fps, reader.width, reader.height, camera.camera_id, moment, f"{video.stem}/{perception.tracker.type}", settings, config_files, models,
```

Replace:

```python
            return review_check(reader, check, context, plate_models, plates.reading, evidence, output, unavailable or "no plate models")
```

with:

```python
            return review_check(reader, check, context, plate_models, plates.reading, evidence, output, unavailable)
```

In `src/tvd/service/runner.py`:

After:

```python
from tvd.service.settings import SettingsFiles
```

add:

```python
from tvd.versions import models_used
```

After:

```python
                "plates": self.plates.model_dump(), "evidence": self.evidence.model_dump(),
            }
```

add:

```python
            plate_models = self.plate_models()
```

After:

```python
                f"{job.job_id}/{self.perception.tracker.type}", settings, {**self.config_files, "camera": camera.version},
```

add:

```python
                models_used(self.detector, self.perception.tracker, plate_models, self.plates_unavailable),
```

Replace:

```python
            return review_check(reader, check, context, self.plate_models(), self.plates.reading, self.evidence, folder, self.plates_unavailable)
```

with:

```python
            return review_check(reader, check, context, plate_models, self.plates.reading, self.evidence, folder, self.plates_unavailable)
```

In `docs/event_schema.md`:

After:

```markdown
| `config_files` | New: the version of each settings file, as `sha256:` and the first 12 hex digits of its hash |
```

add:

```markdown
| `models` | New: the models and software behind the record. `detector` names its weights, their version and its device (`cuda:0` or `cpu`); `tracker` its type and options; `plates` the plate models and their ONNX providers, the first being `CUDAExecutionProvider` on the GPU, or `unavailable` with the reason; `software` the versions of the libraries; `code` the commit of the Docker image, or null |
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `349 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_versions.py tests/tvd/test_tvd_records.py tests/tvd/test_tvd_review.py tests/tvd/test_tvd_runner.py tests/tvd/test_tvd_cli.py src/tvd/versions.py src/tvd/records.py src/tvd/cli.py src/tvd/service/runner.py docs/event_schema.md
git commit -m "Keep the models and software behind each record"
```

### Task 3: A log of JSON lines

**Why:** A service that runs alone needs a log a program can read: each line is one JSON object with a time, a level, an event and that event's fields. It goes to a file that is set aside when it grows, and to standard output, which is what `docker logs` shows. Where and how much to keep is in `configs/service.yaml`.

**Files:**
- Create: `tests/tvd/test_tvd_logs.py`
- Change: `configs/service.yaml`
- Change: `src/tvd/service/settings.py`
- Create: `src/tvd/service/logs.py`

**Interfaces:**
- Consumes: `ServiceSettings` (Part 4).
- Produces: `LoggingSettings(folder, level, max_mb, keep_files)` and `ServiceSettings.logging`; `LOGGER = "tvd.service"`, `JsonFormatter`, `log_event(event, level=INFO, exc_info=False, **fields)` and `setup_logging(settings) -> Path` in `tvd/service/logs.py`.

- [ ] **Step 1: Write the failing test**

A fixture puts the logger back as it was after each test, because `setup_logging` changes a logger that every test shares.

`tests/tvd/test_tvd_logs.py`:

```python
import json
import logging

import pytest
from service_data import settings_for

from tvd.config import ConfigError
from tvd.service.logs import LOGGER, log_event, setup_logging
from tvd.service.settings import LoggingSettings


@pytest.fixture(autouse=True)
def restore_logger():
    yield
    logger = logging.getLogger(LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_service_file_says_where_and_how_much_to_log(tmp_path):
    settings = settings_for(tmp_path).logging
    assert (settings.folder.name, settings.level, settings.max_mb, settings.keep_files) == ("logs", "INFO", 20, 5)


def test_a_wrong_level_is_named(tmp_path):
    from service_data import service_files
    from tvd.service.settings import load_service_settings

    path = service_files(tmp_path)
    path.write_text(path.read_text().replace("level: INFO", "level: LOUD"))
    with pytest.raises(ConfigError, match="logging.level: Input should be 'DEBUG', 'INFO', 'WARNING' or 'ERROR'"):
        load_service_settings(path)


def test_each_event_is_one_json_line_in_the_file_and_on_standard_output(tmp_path, capsys):
    path = setup_logging(LoggingSettings(folder=tmp_path / "logs", level="INFO", max_mb=1, keep_files=2))
    log_event("job_submitted", job_id="20261108-100314-fb185c", camera_id="cam1", estimate_seconds=190.1)
    (line,) = lines(path)
    assert path == tmp_path / "logs" / "service.log" and line["event"] == "job_submitted" and line["level"] == "INFO"
    assert (line["job_id"], line["camera_id"], line["estimate_seconds"]) == ("20261108-100314-fb185c", "cam1", 190.1) and line["time"].endswith("+00:00")
    assert json.loads(capsys.readouterr().out.strip()) == line


def test_an_error_keeps_its_traceback(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="INFO", max_mb=1, keep_files=2))
    try:
        raise RuntimeError("ffmpeg failed")
    except RuntimeError:
        log_event("job_failed", logging.ERROR, job_id="x", exc_info=True)
    (line,) = lines(path)
    assert line["level"] == "ERROR" and "RuntimeError: ffmpeg failed" in line["error"] and "Traceback" in line["error"]


def test_events_below_the_level_are_left_out(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="WARNING", max_mb=1, keep_files=2))
    log_event("job_progress", stage="tracking")
    log_event("worker_error", logging.WARNING, error="database is locked")
    assert [line["event"] for line in lines(path)] == ["worker_error"]


def test_a_full_file_is_rotated_and_only_so_many_are_kept(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="INFO", max_mb=0.001, keep_files=2))  # About 1 kB per file.
    for k in range(100):
        log_event("job_progress", job_id="x", frames_done=k)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["service.log", "service.log.1", "service.log.2"]


def test_starting_twice_does_not_write_each_line_twice(tmp_path):
    settings = LoggingSettings(folder=tmp_path, level="INFO", max_mb=1, keep_files=2)
    setup_logging(settings)
    path = setup_logging(settings)
    log_event("service_started")
    assert len(lines(path)) == 1
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_logs.py -q
```

Expected: `E   ModuleNotFoundError: No module named 'tvd.service.logs'`

- [ ] **Step 3: Write the code**

In `configs/service.yaml`:

After:

```yaml
    orb: 7.0                    #   (camera motion, 127.3 s for 27.2 s), times 1.5: about 17 min at most
```

add:

```yaml
logging:
  folder: outputs/service/logs  # one JSON object per line, never a plate number; also on standard output
  level: INFO                   # DEBUG, INFO, WARNING or ERROR
  max_mb: 20                    # a file this large is set aside, and a new one begun
  keep_files: 5                 # how many files set aside are kept
```

In `src/tvd/service/settings.py`:

After:

```python
from __future__ import annotations

from pathlib import Path
```

add:

```python
from typing import Literal
```

Before:

```python
class ServerSettings(StrictModel):
```

add:

```python
class LoggingSettings(StrictModel):
    folder: Path
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    max_mb: float = Field(gt=0)
    keep_files: int = Field(ge=1)
```

After:

```python
    storage: StorageSettings
    jobs: JobSettings
```

add:

```python
    logging: LoggingSettings
```

`src/tvd/service/logs.py`:

```python
"""The service's log: one JSON object per line, in a file that is rotated, and on standard output for ``docker logs``.

Each line has a time, a level and an event, with that event's fields, such as a job id. No line ever holds a plate number:
events carry ids, states, counts and reasons, never a record.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from tvd.service.settings import LoggingSettings

LOGGER = "tvd.service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname, "event": record.getMessage(), **getattr(record, "fields", {}),
        }
        if record.exc_info:
            line["error"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def log_event(event: str, level: int = logging.INFO, exc_info: bool = False, **fields: Any) -> None:
    """``log_event("job_done", job_id=...)``: one line of the service's log."""
    logging.getLogger(LOGGER).log(level, event, exc_info=exc_info, extra={"fields": fields})


def setup_logging(settings: LoggingSettings) -> Path:
    """Send the service's events to ``<folder>/service.log`` and to standard output, and return the file's path.

    Called when the service starts. Called again, it replaces its handlers, so no line is written twice.
    """
    settings.folder.mkdir(parents=True, exist_ok=True)
    path = settings.folder / "service.log"
    logger = logging.getLogger(LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = JsonFormatter()
    to_file = RotatingFileHandler(path, maxBytes=int(settings.max_mb * 2**20), backupCount=settings.keep_files, encoding="utf-8")
    to_output = logging.StreamHandler(sys.stdout)
    for handler in (to_file, to_output):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(settings.level)
    logger.propagate = False  # The lines are complete here; the root logger would print them a second time.
    return path
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `356 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_logs.py configs/service.yaml src/tvd/service/settings.py src/tvd/service/logs.py
git commit -m "Log the service's events as JSON lines"
```

### Task 4: Every job in the log

**Why:** The events of a job, in order: `job_queued` (or `upload_refused`, with the reason), `job_started`, a `job_stage` at each change of stage, then `job_done` with its counts and how long it took next to its estimate, or `job_failed` with the reason and the traceback. Also `job_interrupted` at a restart, `job_removed` by the retention, `plate_models_restarted` and `worker_error`. Comparing `seconds` with `estimate_seconds` over many jobs shows whether the measured speeds of `service.yaml` still hold. **A log line never holds a plate number**: events carry ids, states, counts and reasons, never a record.

**Files:**
- Change: `tests/tvd/test_tvd_logs.py`
- Create: `tests/tvd/test_tvd_service_events.py`
- Change: `src/tvd/service/logs.py`
- Change: `src/tvd/service/core.py`
- Change: `src/tvd/service/runner.py`
- Change: `src/tvd/service/worker.py`

**Interfaces:**
- Consumes: `log_event` (Task 3); `ReviewService`, `Pipeline`, `Worker` (Part 4).
- Produces: The events above, from `ReviewService.submit`, `run_next`, `remove_old` and its start, from `Pipeline` and from `Worker`. `submit` now calls `_admit`, the old body, so that a refusal can be logged in one place. A traceback goes in its own key, `traceback`.

- [ ] **Step 1: Write the failing test**

pytest's `caplog` catches the events. `test_no_log_line_holds_a_record_or_a_plate` passes already: it guards against a record reaching the log later.

In `tests/tvd/test_tvd_logs.py`:

Replace:

```python
    assert line["level"] == "ERROR" and "RuntimeError: ffmpeg failed" in line["error"] and "Traceback" in line["error"]
```

with:

```python
    assert line["level"] == "ERROR" and "RuntimeError: ffmpeg failed" in line["traceback"] and "Traceback" in line["traceback"]


def test_an_error_field_and_a_traceback_are_both_kept(tmp_path):
    path = setup_logging(LoggingSettings(folder=tmp_path, level="INFO", max_mb=1, keep_files=2))
    try:
        raise RuntimeError("database is locked")
    except RuntimeError:
        log_event("worker_error", logging.ERROR, exc_info=True, error="RuntimeError: database is locked")
    (line,) = lines(path)
    assert line["error"] == "RuntimeError: database is locked" and "Traceback" in line["traceback"]
```

`tests/tvd/test_tvd_service_events.py`:

```python
"""What the service writes in its log, from upload to removal. Events carry ids, states, counts and reasons, never a plate."""

import logging
from datetime import timedelta

import pytest
from service_data import PLATES, settings_for
from test_tvd_runner import OneUse  # noqa: F401  Registers "test-one-use-plate".
from test_tvd_service import Clock, start, submit

from tvd.service.core import ReviewService
from tvd.service.intake import Refused
from tvd.service.logs import LOGGER
from tvd.service.runner import Pipeline
from tvd.service.worker import Worker


@pytest.fixture
def events(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return lambda: [(r.getMessage(), getattr(r, "fields", {})) for r in caplog.records if r.name == LOGGER]


def test_a_job_is_logged_from_its_upload_to_its_records(tmp_path, sample_video, events):
    service = start(tmp_path)
    job = submit(service, sample_video)
    service.run_next()
    logged = events()
    assert [name for name, _ in logged] == ["job_queued", "job_started", "job_stage", "job_stage", "job_stage", "job_done"]
    queued, started, *stages, done = (fields for _, fields in logged)
    assert queued == {"job_id": job.job_id, "camera_id": "test-camera", "video_seconds": 10.0, "estimate_seconds": 22.0, "queue_seconds": 0.0}
    assert started == {"job_id": job.job_id} and [s["stage"] for s in stages] == ["tracking", "rules", "evidence"]
    assert (done["job_id"], done["records"], done["plates"], done["failures"], done["estimate_seconds"]) == (job.job_id, 2, {"unreadable": 2}, [], 22.0)
    assert done["seconds"] >= 0


def test_no_log_line_holds_a_record_or_a_plate(tmp_path, sample_video, events):
    service = start(tmp_path)
    submit(service, sample_video)
    service.run_next()
    assert all("license_plate" not in fields and "text" not in fields and "events" not in fields for _, fields in events())


def test_a_refused_upload_is_logged_with_its_reason(tmp_path, sample_video, events):
    service = start(tmp_path)
    with pytest.raises(Refused):
        submit(service, sample_video, camera_id="cam9")
    assert events() == [("upload_refused", {"camera_id": "cam9", "status": 422, "reason": "Unknown camera 'cam9'. Known cameras: test-camera"})]


def test_a_failed_job_is_logged_with_its_reason(tmp_path, sample_video, events, caplog):
    service = start(tmp_path)
    job = submit(service, sample_video)
    (service.folders.folder(job.job_id) / "video.mp4").unlink()
    service.run_next()
    name, fields = events()[-1]
    failed = [r for r in caplog.records if r.getMessage() == "job_failed"][0]
    assert name == "job_failed" and fields["job_id"] == job.job_id and fields["reason"].startswith("FileNotFoundError: Video not found")
    assert failed.levelname == "ERROR" and failed.exc_info is not None


def test_a_restart_logs_the_jobs_it_interrupted_and_the_retention_logs_what_it_deletes(tmp_path, sample_video, events):
    clock = Clock()
    service = start(tmp_path, clock)
    running, finished = submit(service, sample_video), submit(service, sample_video)
    service.store.start(running.job_id, "2026-11-07T01:00:00+00:00")
    service.close()
    restarted = start(tmp_path, clock)
    restarted.run_next()
    clock.now += timedelta(days=8)
    restarted.remove_old()
    names = [(name, fields.get("job_id")) for name, fields in events()]
    assert ("job_interrupted", running.job_id) in names and ("job_removed", running.job_id) in names and ("job_removed", finished.job_id) in names


def test_plate_models_started_again_are_logged(tmp_path, sample_video, events):
    pipeline = Pipeline(settings_for(tmp_path, PLATES.replace("type: test-plate", "type: test-one-use-plate")).files)
    pipeline.plate_models()
    assert ("plate_models_restarted", {}) in events()


def test_a_worker_error_is_logged(events):
    from test_tvd_worker import Flaky, wait_for

    service = Flaky()
    worker = Worker(service, idle_seconds=0.05)
    worker.start()
    try:
        assert wait_for(lambda: service.calls >= 2, seconds=5)
    finally:
        worker.stop(timeout=5)
    assert ("worker_error", {"error": "RuntimeError: database is locked"}) in events()
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_logs.py tests/tvd/test_tvd_service_events.py -q
```

Expected: The new event tests fail with `assert [] == [...]`: nothing is logged yet. In `test_tvd_logs.py`, the two tests of tracebacks fail: the traceback is still under `error`, where it hides an event's own `error` field.

- [ ] **Step 3: Write the code**

In `src/tvd/service/logs.py`:

Replace:

```python
            line["error"] = self.formatException(record.exc_info)
```

with:

```python
            line["traceback"] = self.formatException(record.exc_info)  # Its own key: an event may have an "error" field.
```

In `src/tvd/service/core.py`:

After:

```python
from __future__ import annotations

import fcntl
import json
```

add:

```python
import logging
```

After:

```python
from tvd.service.jobs import DONE, RUNNING, Job, JobStore
```

add:

```python
from tvd.service.logs import log_event
```

After:

```python
        self.interrupted = self.store.interrupt(as_text(clock()))
```

add:

```python
        for job_id in self.interrupted:
            log_event("job_interrupted", logging.WARNING, job_id=job_id)
```

After:

```python
        """Save, check and queue an upload. ``Refused`` says why it was not queued; then nothing of it is kept."""
```

add:

```python
        try:
            job, waiting = self._admit(upload, filename, camera_id, recorded_at)
        except Refused as refused:
            log_event("upload_refused", logging.WARNING, camera_id=camera_id, status=refused.status, reason=refused.reason)
            raise
        log_event(
            "job_queued", job_id=job.job_id, camera_id=camera_id, video_seconds=job.video_seconds,
            estimate_seconds=job.estimate_seconds, queue_seconds=round(waiting, 1),
        )
        return job

    def _admit(self, upload: BinaryIO, filename: str, camera_id: str, recorded_at: str) -> tuple[Job, float]:
```

Replace:

```python
            self.folders.remove(job_id)
            raise
        return job
```

with:

```python
            self.folders.remove(job_id)
            raise
        return job, waiting
```

After:

```python
        self.store.start(job.job_id, as_text(self.clock()))
```

add:

```python
        log_event("job_started", job_id=job.job_id)
        began = self.clock()
        stages: list[str] = []
```

After:

```python
                self.store.progress(job.job_id, stage, done, total)
```

add:

```python
                if not stages or stages[-1] != stage:
                    stages.append(stage)
                    log_event("job_stage", job_id=job.job_id, stage=stage)
```

Replace:

```python
            self.pipeline.run(job, camera, self.folders.folder(job.job_id), report)
```

with:

```python
            result = self.pipeline.run(job, camera, self.folders.folder(job.job_id), report)
```

Replace:

```python
            self.store.fail(job.job_id, f"{type(error).__name__}: {error}", as_text(self.clock()))
```

with:

```python
            reason = f"{type(error).__name__}: {error}"
            self.store.fail(job.job_id, reason, as_text(self.clock()))
            log_event("job_failed", logging.ERROR, exc_info=True, job_id=job.job_id, reason=reason)
```

After:

```python
            self.store.finish(job.job_id, as_text(self.clock()))
```

add:

```python
            summary = result.summary()
            log_event(
                "job_done", job_id=job.job_id, records=summary["records"], plates=summary["plates"], failures=summary["failures"],
                seconds=round((self.clock() - began).total_seconds(), 1), estimate_seconds=job.estimate_seconds,
            )
```

After:

```python
            self.folders.remove(job_id)
            self.store.delete(job_id)
```

add:

```python
            log_event("job_removed", job_id=job_id)
```

In `src/tvd/service/runner.py`:

Before:

```python
from collections.abc import Callable
```

add:

```python
import logging
```

After:

```python
from tvd.service.cameras import Camera
from tvd.service.jobs import Job
```

add:

```python
from tvd.service.logs import log_event
```

After:

```python
            self._plate_models, self.plates_unavailable = None, f"the plate models did not start: {error}"
```

add:

```python
            log_event("plate_models_unavailable", logging.WARNING, reason=self.plates_unavailable)
```

After:

```python
            if self._plate_models is not None:
                self._plate_models.close()
```

add:

```python
                log_event("plate_models_restarted", logging.WARNING)
```

In `src/tvd/service/worker.py`:

Before:

```python
import threading

from tvd.service.core import ReviewService
```

add:

```python
import logging
```

After:

```python
from tvd.service.core import ReviewService
```

add:

```python
from tvd.service.logs import log_event
```

After:

```python
                self.errors.append(f"{type(error).__name__}: {error}")
```

add:

```python
                log_event("worker_error", logging.ERROR, exc_info=True, error=self.errors[-1])
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `364 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_logs.py tests/tvd/test_tvd_service_events.py src/tvd/service/logs.py src/tvd/service/core.py src/tvd/service/runner.py src/tvd/service/worker.py
git commit -m "Log every job from its upload to its removal"
```

### Task 5: Health

**Why:** `GET /health` says whether the service can do its jobs, and how well. **failing** (status 503): no worker runs, or the storage has less free space than `min_free_gb`: no job will be done. **degraded** (200): jobs are done, but worse: the plate models run on the CPU, as when another user fills the shared GPU, or there are none, or the detector runs on the CPU. **ok** otherwise. The answer also gives the queue, the free space, the models and their versions. Docker will call it to know whether the container is healthy (Task 6). `serve` now starts the log first, then logs `service_started` with the models.

**Files:**
- Change: `tests/tvd/test_tvd_service_settings.py`
- Change: `tests/tvd/test_tvd_service_events.py`
- Change: `tests/tvd/test_tvd_api.py`
- Change: `tests/tvd/test_tvd_cli.py`
- Change: `configs/service.yaml`
- Change: `src/tvd/service/settings.py`
- Change: `src/tvd/service/runner.py`
- Change: `src/tvd/service/core.py`
- Change: `src/tvd/service/api.py`
- Change: `src/tvd/cli.py`

**Interfaces:**
- Consumes: `describe()`, `models_used` (Tasks 1 and 2); `setup_logging`, `log_event` (Tasks 3 and 4); `ReviewService`, `Worker`, `create_app`, `run_serve` (Part 4).
- Produces: `StorageSettings.min_free_gb`; `Pipeline.models()`, which does not start stopped plate models again; `ReviewService.health(worker_running) -> dict` and `ReviewService.started_at`; `GET /health`.

- [ ] **Step 1: Write the failing test**

Two fake plate models show the degraded cases: `test-cpu-plate`, which says it runs on the CPU, and `test-never-starts`, from Part 4.

In `tests/tvd/test_tvd_service_settings.py`:

Replace:

```python
    assert settings.files.cameras.as_posix() == "configs/cameras" and settings.storage.keep_days == 7
```

with:

```python
    assert settings.files.cameras.as_posix() == "configs/cameras" and settings.storage.keep_days == 7 and settings.storage.min_free_gb == 10
```

In `tests/tvd/test_tvd_service_events.py`:

Replace:

```python
from test_tvd_runner import OneUse  # noqa: F401  Registers "test-one-use-plate".
```

with:

```python
from conftest import FakePlateModels
from test_tvd_runner import NeverStarts, OneUse  # noqa: F401  Registers "test-never-starts" and "test-one-use-plate".
```

Before:

```python
from tvd.service.core import ReviewService
```

add:

```python
from tvd.plates import PLATE_MODELS
```

Add to the end of the file:

```python
@PLATE_MODELS.register("test-cpu-plate")
class OnTheCpu(FakePlateModels):
    """Models that had to start on the CPU, as when the GPU is full."""

    def describe(self):
        return {"type": "onnx_worker", "providers": ["CPUExecutionProvider"]}


def health_of(tmp_path, plates=PLATES, worker_running=True, **storage):
    settings = settings_for(tmp_path, plates)
    if storage:
        settings = settings.model_copy(update={"storage": settings.storage.model_copy(update=storage)})
    return ReviewService(settings, Clock()).health(worker_running)


def test_a_service_with_its_worker_and_room_is_healthy(tmp_path):
    report = health_of(tmp_path)
    assert (report["status"], report["problems"], report["worker_running"], report["queue"]) == ("ok", [], True, {"jobs": 0, "seconds": 0.0})
    assert report["started_at"] == "2026-11-07T01:00:00+00:00" and report["storage"]["free_gb"] > 0
    assert report["models"]["plates"] == {"type": "FakePlateModels"} and report["models"]["software"]["numpy"] == "2.4.6"


@pytest.mark.parametrize("change, problem", [
    ({"worker_running": False}, "the worker is not running: no job will be done"),
    ({"min_free_gb": 10**9}, "GB free in the storage, under the 1000000000.0 GB it needs"),
])
def test_a_service_that_cannot_do_jobs_is_failing(tmp_path, change, problem):
    report = health_of(tmp_path, **change)
    assert report["status"] == "failing" and any(problem in p for p in report["problems"])


@pytest.mark.parametrize("plates, problem", [
    (PLATES.replace("type: test-plate", "type: test-cpu-plate"), "the plate models run on the CPU: the GPU may be full"),
    (PLATES.replace("type: test-plate", "type: test-never-starts"), "no plates are read: the plate models did not start: Plate models: no such model"),
])
def test_a_service_that_does_jobs_less_well_is_degraded(tmp_path, plates, problem):
    report = health_of(tmp_path, plates)
    assert report["status"] == "degraded" and report["problems"] == [problem]
```

Add to the end of `tests/tvd/test_tvd_api.py`:

```python
def test_health_is_failing_without_a_worker(client):
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["problems"] == ["the worker is not running: no job will be done"]


def test_health_is_ok_while_the_app_runs_its_worker(service):
    with TestClient(create_app(service)) as client:
        response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"
```

In `tests/tvd/test_tvd_cli.py`:

After:

```python
    assert (started["host"], started["port"]) == ("127.0.0.1", 8040) and isinstance(started["app"], FastAPI)
```

add:

```python
    import json
    import logging

    from tvd.service.logs import LOGGER

    for handler in list(logging.getLogger(LOGGER).handlers):  # So that later tests are not logged in this test's folder.
        logging.getLogger(LOGGER).removeHandler(handler)
        handler.close()
    logging.getLogger(LOGGER).propagate = True
    started_line = json.loads((tmp_path / "storage" / "logs" / "service.log").read_text().splitlines()[-1])
    assert started_line["event"] == "service_started" and started_line["port"] == 8040 and started_line["models"]["tracker"] == {"type": "test-same-id"}
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_service_settings.py tests/tvd/test_tvd_service_events.py tests/tvd/test_tvd_api.py tests/tvd/test_tvd_cli.py -q
```

Expected: `AttributeError: 'StorageSettings' object has no attribute 'min_free_gb'`, `AttributeError: 'ReviewService' object has no attribute 'health'`, `/health` answering `404`, and no log file after `serve`.

- [ ] **Step 3: Write the code**

In `configs/service.yaml`:

After:

```yaml
  max_upload_mb: 4096           # a larger upload is refused while it arrives
```

add:

```yaml
  min_free_gb: 10               # less free space than this, and the health check says the service is failing
```

In `src/tvd/service/settings.py`:

After:

```python
    keep_days: float = Field(gt=0)
    max_upload_mb: int = Field(gt=0)
```

add:

```python
    min_free_gb: float = Field(ge=0)
```

In `src/tvd/service/runner.py`:

After:

```python
from collections.abc import Callable
from pathlib import Path
```

add:

```python
from typing import Any
```

Before:

```python
    def run(self, job: Job, camera: Camera, folder: Path, report: Report) -> ReviewResult:
```

add:

```python
    def models(self) -> dict[str, Any]:
        """The models as they are now, for the health check. Plate models that stopped are reported, not started again."""
        alive = self._plate_models is not None and self._plate_models.is_alive()
        unavailable = self.plates_unavailable if self._plate_models is None else "the plate models stopped; they start again with the next job"
        return models_used(self.detector, self.perception.tracker, self._plate_models if alive else None, unavailable)
```

In `src/tvd/service/core.py`:

After:

```python
import json
import logging
import re
```

add:

```python
import shutil
```

After:

```python
        self._admission = threading.Lock()  # Two uploads at once must not both fit in the same room in the queue.
```

add:

```python
        self.started_at = as_text(clock())
```

Before:

```python
    def describe_cameras(self) -> list[dict[str, Any]]:
```

add:

```python
    def health(self, worker_running: bool) -> dict[str, Any]:
        """``ok``, ``degraded`` or ``failing``, with the problems found, for a person and for Docker's health check.

        ``failing``: no job can be done, because no worker runs or the storage is almost full.
        ``degraded``: jobs are done, but slowly or without plates, as when the GPU is full.
        """
        failing, degraded = [], []
        free_gb = shutil.disk_usage(self.folders.root).free / 2**30
        if not worker_running:
            failing.append("the worker is not running: no job will be done")
        if free_gb < self.settings.storage.min_free_gb:
            failing.append(f"{free_gb:.1f} GB free in the storage, under the {self.settings.storage.min_free_gb:.1f} GB it needs")
        models = self.pipeline.models()
        if models["detector"].get("device") == "cpu":
            degraded.append("the detector runs on the CPU: jobs take much longer")
        plates = models["plates"]
        if "unavailable" in plates:
            degraded.append(f"no plates are read: {plates['unavailable']}")
        elif plates.get("providers", ["CUDAExecutionProvider"])[0] != "CUDAExecutionProvider":
            degraded.append("the plate models run on the CPU: the GPU may be full")
        waiting = self.store.waiting()
        return {
            "status": "failing" if failing else "degraded" if degraded else "ok", "problems": failing + degraded,
            "worker_running": worker_running, "queue": {"jobs": len(waiting), "seconds": round(self.queue_seconds(), 1)},
            "storage": {"free_gb": round(free_gb, 1)}, "models": models, "started_at": self.started_at,
        }
```

In `src/tvd/service/api.py`:

After:

```python
| GET | /cameras | the configured cameras and their rules |
```

add:

```python
| GET | /health | ok or degraded (200), or failing (503), with the problems, the queue, the free space and the models |
```

Replace:

```python
        return service.describe_cameras()

    return app
```

with:

```python
        return service.describe_cameras()

    @app.get("/health")
    def health() -> JSONResponse:
        report = service.health(worker is not None and worker.running)
        return JSONResponse(status_code=503 if report["status"] == "failing" else 200, content=report)

    return app
```

In `src/tvd/cli.py`:

After:

```python
    from tvd.service.core import ReviewService
```

add:

```python
    from tvd.service.logs import log_event, setup_logging
```

After:

```python
    settings = load_service_settings(config_path)
```

add:

```python
    setup_logging(settings.logging)  # Before the service starts, so that the jobs it marks failed are logged.
```

Replace:

```python
    for job_id in service.interrupted:
        print(f"job {job_id} was running when the service stopped: it is marked failed", file=sys.stderr)
```

with:

```python
    log_event("service_started", host=settings.server.host, port=settings.server.port, models=service.pipeline.models())
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `371 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_service_settings.py tests/tvd/test_tvd_service_events.py tests/tvd/test_tvd_api.py tests/tvd/test_tvd_cli.py configs/service.yaml src/tvd/service/settings.py src/tvd/service/runner.py src/tvd/service/core.py src/tvd/service/api.py src/tvd/cli.py
git commit -m "Report the service's health, and log its start"
```

### Task 6: A Docker image with the tested versions

**Why:** The image must run what the tests passed with, nothing newer. So it installs `pyproject.toml`'s extras under `docker/constraints.txt`, which is the server environment frozen; it starts from Python 3.11.18, the server's own; and it holds the models the tests used: the YOLO weights, and the plate models copied from the server's cache, so that it never downloads anything. Their licence is not stated, so they stay out of git, in `docker/models/`. The container runs as user 1000, who owns the job folders on the server, listens on every address inside the container (`configs/service.docker.yaml`), and lets Docker check its health. Building it is in Task 8.

**Files:**
- Create: `tests/tvd/test_tvd_docker.py`
- Create: `docker/Dockerfile`
- Create: `docker/constraints.txt`
- Create: `docker/check_plate_models.py`
- Create: `docker/copy_plate_models.py`
- Create: `.dockerignore`
- Change: `.gitignore`
- Create: `configs/service.docker.yaml`
- Change: `scripts/server.sh`

**Interfaces:**
- Consumes: `GET /health` (Task 5); `configs/service.yaml`, `configs/tvd/plates.yaml` (Parts 3 and 4).
- Produces: `docker/Dockerfile`; `docker/constraints.txt`; `docker/copy_plate_models.py <folder>`, which copies the plate models named in `plates.yaml` from `~/.cache`; `docker/check_plate_models.py`, which loads them while the image is built; `.dockerignore`; `configs/service.docker.yaml`; `scripts/server.sh sync` now also sends `docker/`, `.dockerignore` and `.gitignore`.

- [ ] **Step 1: Write the failing test**

These tests check the files; building the image and running it are Task 8. `test_the_pins_are_the_versions_installed_where_the_tests_run` compares every pin with the environment the tests run in, so it only runs on the server.

`tests/tvd/test_tvd_docker.py`:

```python
"""The Docker image's files. Building and running it is checked on the server: see the plan of Part 5."""

import importlib.util
import os
import subprocess
import sys
from importlib import metadata

import pytest
import yaml
from conftest import ROOT

from tvd.service.settings import load_service_settings

DOCKER = ROOT / "docker"
VERIFIED = {
    "numpy": "2.4.6", "opencv-python": "5.0.0.93", "opencv-python-headless": "5.0.0.93", "torch": "2.15.0+cu132", "ultralytics": "8.4.197",
    "onnxruntime-gpu": "1.31.1", "fast-plate-ocr": "1.1.0", "open-image-models": "0.6.0", "pydantic": "2.14.3", "fastapi": "0.162.0", "uvicorn": "0.54.0",
}


def pins() -> dict[str, str]:
    lines = [line.strip() for line in (DOCKER / "constraints.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    return dict(line.split("==", 1) for line in lines)


def test_the_container_listens_on_all_its_addresses_and_keeps_jobs_in_its_volume():
    inside = load_service_settings(ROOT / "configs" / "service.docker.yaml")
    outside = load_service_settings(ROOT / "configs" / "service.yaml")
    assert (inside.server.host, inside.storage.folder.as_posix(), inside.logging.folder.as_posix()) == ("0.0.0.0", "/data/service", "/data/service/logs")
    changed = {"server": {"host"}, "storage": {"folder"}, "logging": {"folder"}}
    for section in ("version", "storage", "jobs", "logging", "server", "files"):
        a, b = yaml.safe_load(inside.model_dump_json()), yaml.safe_load(outside.model_dump_json())
        a, b = a[section], b[section]
        if isinstance(a, dict):
            a, b = ({k: v for k, v in d.items() if k not in changed.get(section, set())} for d in (a, b))
        assert a == b, section  # Everything else is the same as outside a container.


def test_the_image_pins_the_versions_the_code_was_tested_with():
    pinned = pins()
    assert {name: pinned[name] for name in VERIFIED} == VERIFIED and all(" @ " not in v and "/" not in v for v in pinned.values())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Needs the GPU server's environment, which the pins come from.")
def test_the_pins_are_the_versions_installed_where_the_tests_run():
    different = {name: (version, metadata.version(name)) for name, version in pins().items() if metadata.version(name) != version}
    assert different == {}


def test_only_what_the_image_needs_is_sent_to_docker():
    rules = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert rules[0] == "*" and set(rules[1:]) >= {"!docker/", "!src/", "!configs/", "!pyproject.toml", "!weights/yolo26m.pt"}
    assert not [r for r in rules if r.startswith("!") and r.split("/")[0].lstrip("!") in ("data", "outputs", "notebooks")]


def test_the_image_has_the_cuda_library_onnx_runtime_takes_from_the_server_s_own_cuda():
    # Without cuRAND the plate models fall back to the CPU, and the health check says "degraded".
    assert "pip install --no-deps nvidia-curand==10.4.2.66" in (DOCKER / "Dockerfile").read_text()


def test_the_image_runs_as_the_storage_s_owner_and_checks_its_own_health():
    dockerfile = (DOCKER / "Dockerfile").read_text()
    assert "FROM python:3.11.18-slim-bookworm" in dockerfile and "pip install --no-deps boxmot==26.0.0" in dockerfile
    assert "USER 1000:1000" in dockerfile and "HEALTHCHECK" in dockerfile and "/health" in dockerfile
    assert "COPY docker/models /app/home/.cache" in dockerfile and "RUN python docker/check_plate_models.py" in dockerfile
    assert 'CMD ["python", "-m", "tvd.cli", "serve", "--config", "configs/service.docker.yaml"]' in dockerfile


def copy_models(home, target):
    return subprocess.run([sys.executable, str(DOCKER / "copy_plate_models.py"), str(target)], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "HOME": str(home)})


def test_the_plate_models_the_tests_used_go_into_the_image(tmp_path):
    for folder, name in (("open-image-models/yolo-v9-t-640-license-plate-end2end", "model.onnx"), ("fast-plate-ocr/cct-s-v2-global-model", "ocr.onnx")):
        (tmp_path / "home" / ".cache" / folder).mkdir(parents=True)
        (tmp_path / "home" / ".cache" / folder / name).write_bytes(b"weights")
    assert copy_models(tmp_path / "home", tmp_path / "models").returncode == 0
    copied = sorted(p.relative_to(tmp_path / "models").as_posix() for p in (tmp_path / "models").rglob("*.onnx"))
    assert copied == ["fast-plate-ocr/cct-s-v2-global-model/ocr.onnx", "open-image-models/yolo-v9-t-640-license-plate-end2end/model.onnx"]


def test_plate_models_never_downloaded_here_are_named(tmp_path):
    done = copy_models(tmp_path, tmp_path / "models")
    assert done.returncode != 0 and "open-image-models/yolo-v9-t-640-license-plate-end2end" in done.stderr


def test_the_plate_models_stay_out_of_git():
    assert "docker/models/" in (ROOT / ".gitignore").read_text().splitlines()
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_docker.py -q
```

Expected: `9 failed`: the files do not exist yet (`FileNotFoundError`), `.gitignore` does not keep `docker/models/` out, and `copy_plate_models.py` cannot be found.

- [ ] **Step 3: Write the code**

`scripts/server.sh` comes last: until it changes, `sync` does not send `docker/`, `.dockerignore` or `.gitignore`, so the tests cannot pass on the server.

`docker/Dockerfile`:

```dockerfile
# The review service, with the versions its tests passed with on ai_server_2. Build it from the project folder:
#   scripts/deploy.sh build <commit>
FROM python:3.11.18-slim-bookworm

# The full OpenCV package, which Ultralytics needs, links to these two libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app/home YOLO_CONFIG_DIR=/app/home/ultralytics
WORKDIR /app

# The libraries first, in a layer of their own: they change far less often than the code, so a new build reuses it.
# PyTorch's CUDA 13.2 build comes from PyTorch's own index. BoxMOT is installed without its dependencies, as on the
# server, because it pins an older OpenCV (see pyproject.toml).
COPY docker/constraints.txt docker/constraints.txt
COPY pyproject.toml README.md ./
RUN mkdir -p src/tvd src/traffic_violation_detection \
 && touch src/tvd/__init__.py src/traffic_violation_detection/__init__.py \
 && pip install --extra-index-url https://download.pytorch.org/whl/cu132 -c docker/constraints.txt ".[detection,tracking,plates,service]" \
 && pip install --no-deps boxmot==26.0.0

# ONNX Runtime's CUDA provider needs cuRAND. On the server it finds the one of the system's CUDA 12.4; the image has no
# system CUDA, so it gets the cuRAND of CUDA 13.2 from pip, as PyTorch gets its own CUDA libraries.
RUN pip install --no-deps nvidia-curand==10.4.2.66

# The models the tests used, inside the image: it runs the same models wherever it runs, and never downloads them.
# scripts/deploy.sh build copies the plate models from the server's cache to docker/models first.
COPY weights/yolo26m.pt weights/yolo26m.pt
COPY configs configs
COPY docker/models /app/home/.cache
COPY docker/check_plate_models.py docker/check_plate_models.py
RUN python docker/check_plate_models.py

# The code last: a change to it rebuilds only this layer.
COPY src src
RUN pip install --no-deps . && chown -R 1000:1000 /app/home

# The commit the image was built from: every record and the health check name it.
ARG TVD_COMMIT
ENV TVD_COMMIT=${TVD_COMMIT}

# The user that owns the job folders on the server, so that both can read and write them.
USER 1000:1000
EXPOSE 8040
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8040/health', timeout=8)"]
CMD ["python", "-m", "tvd.cli", "serve", "--config", "configs/service.docker.yaml"]
```

`docker/constraints.txt` is not written by hand: it is every package of the environment where the tests pass, frozen. One line, from `pip`, is a path on the server instead of a version, so it is replaced by its version:

```bash
{ echo "# Every package's version in the tvd environment of ai_server_2, where the tests pass: pip freeze --exclude-editable."
  echo "# The image installs pyproject.toml's extras under these constraints, so that it gets the versions that were tested."
  scripts/server.sh run 'pip freeze --exclude-editable | grep -v " @ "; echo "packaging==$(python -c "import importlib.metadata as m; print(m.version(\"packaging\"))")"' | sort -f
} > docker/constraints.txt
grep -E "^(numpy|torch|onnxruntime-gpu)==" docker/constraints.txt
```

Expected: 183 lines, among them `numpy==2.4.6`, `onnxruntime-gpu==1.31.1` and `torch==2.15.0+cu132`.

`docker/check_plate_models.py`:

```python
"""Load the plate models copied into the image, on the CPU (building has no GPU), so that a missing file fails the build.

The libraries find them in their cache, $HOME/.cache, and download nothing.
"""

import yaml
from fast_plate_ocr import LicensePlateRecognizer
from open_image_models import create_detector

options = yaml.safe_load(open("configs/tvd/plates.yaml"))["models"]["options"]
create_detector(options["detector_model"], providers=["CPUExecutionProvider"])
LicensePlateRecognizer(options["ocr_model"], providers=["CPUExecutionProvider"])
print("plate models ready:", options["detector_model"], options["ocr_model"])
```

`docker/copy_plate_models.py`:

```python
"""Copy the plate models named in configs/tvd/plates.yaml from this machine's cache, where the tests used them, to a folder.

``python docker/copy_plate_models.py docker/models``: the image takes them from there (docker/Dockerfile). They are not in git:
their licence is not stated. A model never downloaded here is named, with how to get it.
"""

import shutil
import sys
from pathlib import Path

import yaml

options = yaml.safe_load(Path("configs/tvd/plates.yaml").read_text())["models"]["options"]
target = Path(sys.argv[1])
cache = Path.home() / ".cache"
for folder in (f"open-image-models/{options['detector_model']}", f"fast-plate-ocr/{options['ocr_model']}"):
    source = cache / folder
    if not source.is_dir():
        sys.exit(f"No plate model in {source}: run tests/tvd/test_tvd_onnx_plates.py once on this machine, which downloads it.")
    shutil.copytree(source, target / folder, dirs_exist_ok=True)
    print(f"copied {folder}")
```

`.dockerignore`:

```text
# Only what the image needs is sent to Docker: never data, outputs, notebooks or the private labels.
*
!docker/
!src/
!configs/
!pyproject.toml
!README.md
!weights/yolo26m.pt
**/__pycache__
**/*.egg-info
```

In `.gitignore`:

After:

```text
# Model weights / exported models
weights/
```

add:

```text
# The plate models copied into the Docker image by scripts/deploy.sh build: their licence is not stated.
docker/models/
```

`configs/service.docker.yaml`:

```yaml
# The review service inside its Docker container: configs/service.yaml, but for the container (see docker/Dockerfile).
# Paths are relative to /app, the image's working folder. /data/service is the volume that scripts/deploy.sh mounts.
version: 1
storage:
  folder: /data/service        # each job gets <folder>/<job id>/: outputs/service of the project folder, on the server
  keep_days: 7                  # a finished job's folder and record are deleted after this
  max_upload_mb: 4096           # a larger upload is refused while it arrives
  min_free_gb: 10               # less free space than this, and the health check says the service is failing
jobs:
  deadline_minutes: 120         # results within 2 hours of upload, including the wait in the queue
  seconds_per_video_second:     # work per second of video, by the camera's motion type: measured on ai_server_2 for
    fixed: 2.2                  #   Part 4 (1080p, 38.7 s for 27.2 s of video), times 1.5 for margin: about 55 min at most
    orb: 7.0                    #   (camera motion, 127.3 s for 27.2 s), times 1.5: about 17 min at most
logging:
  folder: /data/service/logs   # one JSON object per line, never a plate number; also on standard output, for docker logs
  level: INFO                   # DEBUG, INFO, WARNING or ERROR
  max_mb: 20                    # a file this large is set aside, and a new one begun
  keep_files: 5                 # how many files set aside are kept
server:
  host: 0.0.0.0                 # every address of the container: Docker publishes the port on 127.0.0.1 of the server only
  port: 8040                    # 8000 is taken on ai_server_2
files:
  cameras: configs/cameras      # one <camera id>.yaml per camera
  perception: configs/tvd/perception.yaml
  rules: configs/tvd/rules.yaml
  plates: configs/tvd/plates.yaml
  evidence: configs/tvd/evidence.yaml
```

In `scripts/server.sh`:

Replace:

```bash
#   scripts/server.sh sync             make the server's src/, tests/, configs/, scripts/ and pyproject.toml the same as here
```

with:

```bash
#   scripts/server.sh sync             make the server's src/, tests/, configs/, scripts/, docker/, pyproject.toml, .dockerignore and .gitignore the same as here
```

Replace:

```bash
            src tests configs scripts pyproject.toml "$HOST:$REMOTE_DIR/"
```

with:

```bash
            src tests configs scripts docker pyproject.toml .dockerignore .gitignore "$HOST:$REMOTE_DIR/"
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `380 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_docker.py docker/Dockerfile docker/constraints.txt docker/check_plate_models.py docker/copy_plate_models.py .dockerignore .gitignore configs/service.docker.yaml scripts/server.sh
git commit -m "Build the service into a Docker image with the tested versions"
```

### Task 7: One script to deploy

**Why:** Every command of the deployment is in `scripts/deploy.sh`, run in the project folder on the server: `build`, `start`, `stop`, `status`, `logs` and `health`. `start` runs the container on the GPU, publishes its port on `127.0.0.1` of the server only, mounts `outputs/service` for its jobs, and asks Docker to start it again after a crash or a reboot (`--restart unless-stopped`). With `TVD_DRY_RUN` set, it prints each command instead of running it, which is how the tests check the commands without Docker.

**Files:**
- Create: `tests/tvd/test_tvd_deploy.py`
- Create: `scripts/deploy.sh`

**Interfaces:**
- Consumes: `docker/Dockerfile`, `docker/copy_plate_models.py` (Task 6).
- Produces: `scripts/deploy.sh build <commit> | start | stop | status | logs [lines] | health`; the image `tvd-service:<commit>`, also tagged `latest`; the container `tvd-service`.

- [ ] **Step 1: Write the failing test**

`tests/tvd/test_tvd_deploy.py`:

```python
"""scripts/deploy.sh, with TVD_DRY_RUN set: it prints each docker command instead of running it."""

import os
import subprocess

from conftest import ROOT

SCRIPT = ROOT / "scripts" / "deploy.sh"


def deploy(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *arguments], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "TVD_DRY_RUN": "1"})


def test_the_image_gets_the_tested_plate_models_and_is_named_after_its_commit():
    done = deploy("build", "6b1a1ba")
    assert done.returncode == 0
    assert done.stdout.strip().split("\n") == [
        "python docker/copy_plate_models.py docker/models",
        "docker build -f docker/Dockerfile --build-arg TVD_COMMIT=6b1a1ba -t tvd-service:6b1a1ba -t tvd-service:latest .",
    ]


def test_the_service_runs_on_the_gpu_on_this_machine_only_and_is_restarted_by_docker():
    command = deploy("start").stdout.split("\n")[-2].split()
    assert command[:4] == ["docker", "run", "-d", "--name"] and command[4] == "tvd-service" and command[-1] == "tvd-service:latest"
    flags = " ".join(command)
    assert "--gpus all" in flags and "--restart unless-stopped" in flags and "-p 127.0.0.1:8040:8040" in flags
    assert f"-v {ROOT}/outputs/service:/data/service" in flags


def test_a_build_needs_its_commit_and_an_unknown_command_is_refused():
    assert deploy("build").returncode != 0
    unknown = deploy("deploy")
    assert unknown.returncode == 2 and "usage: scripts/deploy.sh" in unknown.stderr
```

- [ ] **Step 2: Run the tests and see them fail**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd/test_tvd_deploy.py -q
```

Expected: `3 failed`: `bash: scripts/deploy.sh: No such file or directory`, so the return codes are 127.

- [ ] **Step 3: Write the code**

Make the script executable: `chmod +x scripts/deploy.sh`.

`scripts/deploy.sh`:

```bash
#!/usr/bin/env bash
# The review service in Docker, on the machine this runs on (ai_server_2). Run it in the project folder there:
#   scripts/server.sh run scripts/deploy.sh <command>
#
#   build <commit>   copy the plate models the tests used into docker/models, then build the image
#                    tvd-service:<commit> from docker/Dockerfile, and tag it tvd-service:latest
#   start            run tvd-service:latest: on the GPU, on port 8040 of this machine only, with its jobs in
#                    outputs/service, and started again by Docker after a crash or a reboot
#   stop             stop it and remove the container; the jobs stay in outputs/service
#   status           the container, its image and its health
#   logs [lines]     the last lines of its log (50 by default)
#   health           what GET /health says
#
# With TVD_DRY_RUN set, each docker command is printed instead of run.
set -euo pipefail
NAME=tvd-service
IMAGE=tvd-service
STORAGE="$PWD/outputs/service"

run() {
    if [ -n "${TVD_DRY_RUN:-}" ]; then echo "$*"; else "$@"; fi
}

case "${1:-}" in
    build)
        commit="${2:?the commit to name the image after, such as $(git rev-parse --short HEAD 2>/dev/null || echo 6b1a1ba)}"
        run python docker/copy_plate_models.py docker/models
        run docker build -f docker/Dockerfile --build-arg "TVD_COMMIT=$commit" -t "$IMAGE:$commit" -t "$IMAGE:latest" .
        ;;
    start)
        # The jobs' folder belongs to this user, as does the container's user (1000): both can read and write it.
        run mkdir -p "$STORAGE"
        run docker run -d --name "$NAME" --gpus all --restart unless-stopped -p 127.0.0.1:8040:8040 -v "$STORAGE:/data/service" "$IMAGE:latest"
        ;;
    stop)
        run docker stop "$NAME"
        run docker rm "$NAME"
        ;;
    status)
        run docker ps -a --filter "name=^$NAME\$" --format "{{.Names}} {{.Image}} {{.Status}}"
        ;;
    logs)
        run docker logs --tail "${2:-50}" "$NAME"
        ;;
    health)
        run curl -s localhost:8040/health
        ;;
    *)
        echo "usage: scripts/deploy.sh build <commit> | start | stop | status | logs [lines] | health" >&2
        exit 2
        ;;
esac
```

- [ ] **Step 4: Run every test of the package and see them pass**

```bash
scripts/server.sh sync && scripts/server.sh test tests/tvd -q
```

Expected: `383 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/tvd/test_tvd_deploy.py scripts/deploy.sh
git commit -m "Deploy the service's container with one script"
```

### Task 8: Deploy, and check it end to end

**Why:** The tests check every piece; this task checks the whole, as it will run: the image built on the server, the container on the GPU, a video sent with curl, the records and clips it gives, its log, and Docker starting it again after a crash.

This is the deployment itself: the image is built on the server, and the service runs in its container. Every command runs in the project folder on the server.

**Build the image**, named after your commit. The plate models come from the server's cache, where the tests downloaded them; the libraries take about 4 minutes the first time, and seconds after that, because Docker keeps that layer:

```bash
scripts/server.sh run "scripts/deploy.sh build $(git rev-parse --short HEAD)" | tail -3
scripts/server.sh run 'docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep tvd-service'
```

Expected: `naming to docker.io/library/tvd-service:latest done`, and two tags of about 7.9 GB, which are one image.

**Start it, and ask how it is.** Loading the models takes about half a minute; until the first health check passes, Docker says `health: starting`:

```bash
scripts/server.sh run 'scripts/deploy.sh start; sleep 45; scripts/deploy.sh status; scripts/deploy.sh health'
```

Expected: `tvd-service tvd-service:latest Up ... (healthy)`, then `"status":"ok","problems":[]`, with `"device":"cuda:0"` for the detector, `"providers":["CUDAExecutionProvider","CPUExecutionProvider"]` for the plates, and your commit in `"code"`. If another user's job fills the GPU, the plates may say `CPUExecutionProvider` first and the status `degraded`: that is the health check doing its work.

**Send a video, and follow it**, as in Part 4, but now to the container:

```bash
scripts/server.sh run 'curl -s -F video=@data/traffic_video_modified.mp4 -F camera_id=traffic_video_modified -F recorded_at=2026-11-08T19:30:00+07:00 localhost:8040/jobs'
scripts/server.sh run 'curl -s localhost:8040/jobs/<job id>'
```

Expected: the job, then, after about 3 minutes, `"state":"done"`. Then the records, which now name their models, and a clip compared with Notebook 08's:

```bash
scripts/server.sh run 'curl -s localhost:8040/jobs/<job id>/records | python -c "import json, sys; r = json.load(sys.stdin)[\"events\"]; print(len(r), [e[\"license_plate\"][\"status\"] for e in r], r[0][\"models\"][\"code\"], r[0][\"models\"][\"detector\"][\"device\"])"'
scripts/server.sh run 'curl -s localhost:8040/jobs/<job id>/files/clips/evt-<job id>-red_light-f00317-t23.mp4 | cmp - outputs/08_end_to_end_baseline/traffic_video_modified/violations/traffic_video_modified_violation_clips/evt-traffic_video_modified-main-red_light-f00317-t23.mp4 && echo "clip: same"'
```

Expected: `2 ['read', 'read'] <your commit> cuda:0`, and `clip: same`: inside Docker too, the clips are Notebook 08's, byte for byte.

**Read its log**, one JSON object per line, never a plate:

```bash
scripts/server.sh run 'scripts/deploy.sh logs 400 2>&1 | grep "^{" | tail -2'
```

uvicorn writes its own lines, in plain text, to standard error: `2>&1` lets `grep` keep only the service's JSON lines. Expected: the job's last `job_stage` line, then `job_done` with `"records": 2`, `"plates": {"read": 2}` and its `seconds` next to its `estimate_seconds`. The same lines are in `outputs/service/logs/service.log`.

**Check that Docker starts it again by itself.** Kill its main process from the server, as a crash would:

```bash
scripts/server.sh run 'kill -9 $(docker inspect --format "{{.State.Pid}}" tvd-service); sleep 45; docker inspect --format "{{.RestartCount}}" tvd-service; curl -s localhost:8040/jobs/<job id> | grep -o "\"state\":\"[a-z]*\""'
```

Expected: `1`, then `"state":"done"`: Docker started the service again, and the service still knows the job. Docker does the same after the server reboots, because of `--restart unless-stopped`.

**Check the whole suite once more**, the notebooks' tests included:

```bash
scripts/server.sh test -q
```

Expected: `852 passed`.

The service now runs on the server. To reach it from your laptop: `ssh -N -L 8040:localhost:8040 20.0.14.182`, then `http://localhost:8040`. To stop it: `scripts/server.sh run scripts/deploy.sh stop`. It holds about 0.5 GB of the shared GPU when idle.

---

## What building it taught

The image was first built in three commits, `a5db9a9`, `86f92ac` and `2b7d514`. Task 6 above is the third; the first two had a fault each, and both are worth knowing.

- **Downloading models while building is fragile.** The first image downloaded the plate models in its build, and the build failed: the models' host answered `HTTP 503 Service Unavailable`. Copying the files the tests used, from the server's cache, removed the dependency on that host, and made sure the image holds the very files that were tested.
- **A library the server has, and the image does not.** In the second image, the health check said `degraded` at once: the plate models ran on the CPU. ONNX Runtime's CUDA provider needs cuRAND (`libcurand.so.10`). The `tvd` environment has no cuRAND of its own: on the server, ONNX Runtime quietly loads the one of the system's CUDA 12.4. The image has no system CUDA, so it now installs the cuRAND of CUDA 13.2 from pip (`nvidia-curand==10.4.2.66`). Without the health check, plates would only have been read more slowly, and nobody would have known why.

## When Part 5 is done

- [ ] `scripts/server.sh test tests/tvd -q` gives `383 passed`, and `scripts/server.sh test -q` gives `852 passed`.
- [ ] `scripts/server.sh run scripts/deploy.sh status` says `(healthy)`, and `health` says `"status":"ok"` with the plate models on `CUDAExecutionProvider`.
- [ ] A video sent with curl gives Notebook 08's clips, and its records name the commit of the image.
- [ ] Killed, the service is started again by Docker, and still knows its jobs.
- [ ] `git status` shows nothing under `docker/models/` or `outputs/`.

**Exercises, if you want them** (not checked while writing the plan):

- **JSON access lines**: uvicorn still writes its access lines as plain text. Give uvicorn a logging configuration that uses `JsonFormatter`, so that `docker logs` holds JSON only.
- **A smaller image**: the image is about 7.9 GB, mostly PyTorch and its CUDA libraries. Try a multi-stage build, and measure what is left.
- **Estimates that learn**: from the `job_done` lines of the log, compute the real seconds per second of video of each camera, and compare them with `seconds_per_video_second` in `service.yaml`.

**Next:** the five parts of the spec are done. What the spec leaves for later: live camera streams, once uploads are stable; the review interface and the database of confirmed violations, in your backend; and, if the service leaves the demo, authentication and a database that is not a file.
