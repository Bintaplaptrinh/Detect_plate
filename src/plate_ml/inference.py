"""Recognize one image URL using the locally registered glyph-filter release."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

from .glyph_quality import GlyphQualityConfig, QualityFilteredSegmenter
from .pipeline import PlateRecognizer
from .registry import ModelRegistry
from .research_experiment import recognizer_for


MAX_IMAGE_BYTES = 20 * 1024 * 1024


def read_image_url(image_url, timeout=30):
    """Download an HTTP(S) image into memory with a timeout and size limit."""
    parsed = urlsplit(image_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("image_url must be an HTTP or HTTPS URL")
    request = Request(image_url, headers={"User-Agent": "PlateDetect/0.1"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image download exceeds the 20 MiB limit")
    if not data:
        raise ValueError("Image URL returned an empty response")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("URL did not return a decodable image")
    return image


def load_recognizer(registry_root, version="v1"):
    """Load the same self-contained release as the glyph-filter experiment."""
    registry = ModelRegistry(registry_root)
    try:
        saved, _ = registry.load("research_glyph_filter", version, trusted=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Missing model release: {registry.root / 'Registry' / 'research_glyph_filter' / version}. "
            "Restore the local release described in docs/reproduction.md."
        ) from exc
    base = saved["base_bundle"]
    baseline = PlateRecognizer.from_bundle(base["baseline_bundle"], registry.root / "RuntimeAssets")
    recognizer = recognizer_for(base["selection_config"], baseline, base["verifier"])
    recognizer.segmenter = QualityFilteredSegmenter(
        recognizer.segmenter, GlyphQualityConfig(**saved["quality_config"])
    )
    return recognizer


def run(image_url, device="cpu", *, registry_root="Registration", version="v1"):
    """Return the best plate and request latency, including model load and download."""
    if device != "cpu":
        raise ValueError(
            "GPU inference is not supported by the current OpenCV/scikit-learn HOG/SVM pipeline. "
            "Use --cpu."
        )
    start = perf_counter()
    recognizer = load_recognizer(Path(registry_root), version)
    image = read_image_url(image_url)
    inference_start = perf_counter()
    _, predictions = recognizer.recognize(image)
    end = perf_counter()
    plate = predictions[0]["decoded"]["text"] if predictions else ""
    return {
        "plate": plate,
        "response_ms": round((end - start) * 1000, 3),
        "inference_ms": round((end - inference_start) * 1000, 3),
        "device": "cpu",
        "status": "recognized" if plate else "no_plate",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recognize a plate from an image URL.")
    parser.add_argument("image_url", help="HTTP(S) URL of the vehicle image")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--cpu", dest="device", action="store_const", const="cpu", help="Run on CPU (default)")
    devices.add_argument("--gpu", dest="device", action="store_const", const="gpu", help="Reserved; current HOG/SVM pipeline does not support GPU")
    parser.set_defaults(device="cpu")
    parser.add_argument("--registry", default="Registration", help="Local trusted model registry (default: Registration)")
    parser.add_argument("--version", default="v1", help="research_glyph_filter version (default: v1)")
    args = parser.parse_args(argv)
    try:
        result = run(args.image_url, args.device, registry_root=args.registry, version=args.version)
    except (OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
