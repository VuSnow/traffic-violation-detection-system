"""Download the plate models named in configs/tvd/plates.yaml into the image, on the CPU: building has no GPU."""

import yaml
from fast_plate_ocr import LicensePlateRecognizer
from open_image_models import create_detector

options = yaml.safe_load(open("configs/tvd/plates.yaml"))["models"]["options"]
create_detector(options["detector_model"], providers=["CPUExecutionProvider"])
LicensePlateRecognizer(options["ocr_model"], providers=["CPUExecutionProvider"])
print("plate models ready:", options["detector_model"], options["ocr_model"])
