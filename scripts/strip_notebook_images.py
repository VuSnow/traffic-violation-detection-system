"""Git clean filter: removes image outputs from a notebook, so that they stay in the local copy.

Reads a notebook on stdin and writes it to stdout, in the format nbformat uses. Text outputs are kept.
See "Image outputs stay local" in the README for how to enable it.
"""

import json
import sys


def strip_images(notebook: dict) -> dict:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = [
                output
                for output in cell.get("outputs", [])
                if not any(mime.startswith("image/") for mime in output.get("data", {}))
            ]
    return notebook


if __name__ == "__main__":
    # The bytes are decoded explicitly, since a filter must not depend on the locale.
    notebook = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    text = json.dumps(strip_images(notebook), indent=1, sort_keys=True, ensure_ascii=False, separators=(",", ": "))
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
