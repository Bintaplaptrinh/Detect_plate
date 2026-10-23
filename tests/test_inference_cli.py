from io import BytesIO
import json
from urllib.error import URLError

import cv2
import numpy as np
import pytest

from plate_ml import inference


def test_download_decodes_real_image(monkeypatch):
    image = np.full((40, 80, 3), (10, 80, 200), np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    requests = []

    def open_image(request, timeout):
        requests.append((request.full_url, timeout))
        return BytesIO(encoded.tobytes())

    monkeypatch.setattr(inference, "urlopen", open_image)
    assert np.array_equal(inference.read_image_url("https://example.test/car.png"), image)
    assert requests == [("https://example.test/car.png", 30)]


@pytest.mark.parametrize("data, message", [
    (b"", "empty response"),
    (b"<html>not an image</html>", "decodable image"),
    (b"x" * 101, "20 MiB"),
])
def test_invalid_downloads(monkeypatch, data, message):
    monkeypatch.setattr(inference, "MAX_IMAGE_BYTES", 100)
    monkeypatch.setattr(inference, "urlopen", lambda *a, **kw: BytesIO(data))
    with pytest.raises(ValueError, match=message):
        inference.read_image_url("https://example.test/car.png")


@pytest.mark.parametrize("url", ["file:///tmp/car.png", "ftp://example.test/car.png", "https://", "car.png"])
def test_invalid_url(url):
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        inference.read_image_url(url)


@pytest.mark.parametrize("predictions, plate, status", [
    ([{"decoded": {"text": "29A12345"}}, {"decoded": {"text": "30B54321"}}], "29A12345", "recognized"),
    ([], "", "no_plate"),
    ([{"decoded": {"text": ""}}], "", "no_plate"),
])
def test_run_reports_best_plate_and_latency(monkeypatch, predictions, plate, status):
    class Recognizer:
        def recognize(self, image):
            assert image == "decoded image"
            return None, predictions

    monkeypatch.setattr(inference, "load_recognizer", lambda *a: Recognizer())
    monkeypatch.setattr(inference, "read_image_url", lambda *a: "decoded image")
    times = iter([1.0, 1.5, 1.625])
    monkeypatch.setattr(inference, "perf_counter", lambda: next(times))
    result = inference.run("https://example.test/car.png")
    assert result == dict(plate=plate, status=status, response_ms=625.0, inference_ms=125.0, device="cpu")


def test_gpu_fails_before_loading_or_downloading(capsys):
    assert inference.main(["https://example.test/car.png", "--gpu"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "GPU inference is not supported" in captured.err


def test_cpu_gpu_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        inference.main(["https://example.test/car.png", "--cpu", "--gpu"])
    assert exc.value.code == 2


def test_cli_json_output(monkeypatch, capsys):
    expected = dict(plate="29A12345", response_ms=12.3)
    monkeypatch.setattr(inference, "run", lambda *a, **kw: expected)
    assert inference.main(["https://example.test/car.png", "--cpu"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_network_error_has_nonzero_exit(monkeypatch, capsys):
    def fail(*a, **kw):
        raise URLError("timed out")

    monkeypatch.setattr(inference, "run", fail)
    assert inference.main(["https://example.test/car.png"]) == 1
    assert "timed out" in capsys.readouterr().err


def test_missing_release_explains_setup(tmp_path):
    with pytest.raises(ValueError, match="docs/reproduction.md"):
        inference.load_recognizer(tmp_path / "Registration")
