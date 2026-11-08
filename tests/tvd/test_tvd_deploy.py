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
