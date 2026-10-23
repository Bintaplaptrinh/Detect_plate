"""Download a pinned OpenCV baseline; writes a provenance receipt and license.

This Russian-plate cascade is a transfer baseline. Validate on VNLP before use.
Run explicitly once; library inference and notebooks never fetch network assets.
"""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import json


def fetch(destination: Path):
    base = "https://raw.githubusercontent.com/opencv/opencv/4.13.0/"
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name, relative in (
        ("haarcascade_russian_plate_number.xml", "data/haarcascades/haarcascade_russian_plate_number.xml"),
        ("LICENSE.opencv.txt", "LICENSE"),
    ):
        url = base + relative
        with urlopen(url, timeout=60) as response:
            data = response.read()
        (destination / name).write_bytes(data)
        receipt[name] = dict(url=url, sha256=sha256(data).hexdigest())
    (destination / "provenance.json").write_text(json.dumps(dict(opencv_tag="4.13.0", files=receipt), indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(fetch(Path(__file__).resolve().parents[1] / "assets" / "cascades"), indent=2))
