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
