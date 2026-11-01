"""Tests for scripts/strip_notebook_images.py, run the way git runs it: bytes on stdin and stdout."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "strip_notebook_images.py"

NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": "# Tiêu đề"},
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "source": "print('x')",
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": "precision 13/13\n"},
                {"output_type": "display_data", "metadata": {}, "data": {"image/jpeg": "AAAA", "text/plain": "<Figure>"}},
                {"output_type": "display_data", "metadata": {}, "data": {"image/png": "BBBB"}},
                {"output_type": "execute_result", "execution_count": 1, "metadata": {}, "data": {"text/plain": "42"}},
            ],
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def run_filter(data: bytes) -> bytes:
    return subprocess.run([sys.executable, str(SCRIPT)], input=data, capture_output=True, check=True).stdout


def test_images_are_removed_and_text_outputs_kept():
    out = json.loads(run_filter(json.dumps(NOTEBOOK).encode()))
    outputs = out["cells"][1]["outputs"]
    assert [o["output_type"] for o in outputs] == ["stream", "execute_result"]
    assert b"image/" not in run_filter(json.dumps(NOTEBOOK).encode())


def test_filter_is_idempotent():
    once = run_filter(json.dumps(NOTEBOOK).encode())
    assert run_filter(once) == once


def test_non_ascii_text_survives():
    assert "Tiêu đề".encode() in run_filter(json.dumps(NOTEBOOK).encode())


def test_notebook_without_images_is_unchanged_in_content():
    plain = {**NOTEBOOK, "cells": [NOTEBOOK["cells"][0]]}
    assert json.loads(run_filter(json.dumps(plain).encode())) == plain
