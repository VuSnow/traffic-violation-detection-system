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
