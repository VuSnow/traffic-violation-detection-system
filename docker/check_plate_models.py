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
