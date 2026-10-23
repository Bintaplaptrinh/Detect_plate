from pathlib import Path
import cv2
import numpy as np
from plate_ml.config import DetectionConfig
from plate_ml.detection import PlateDetector
from plate_ml.features import HOGExtractor
from plate_ml.models import make_svm
from plate_ml.pipeline import PlateRecognizer
from plate_ml.registry import ModelRegistry


def test_bundled_cascade_and_relocatable_pipeline(tmp_path):
    cascade = Path(__file__).resolve().parents[1] / "assets" / "cascades" / "haarcascade_russian_plate_number.xml"
    images, labels = [], []
    for char in "2Z":
        for thickness in (1, 2, 3):
            image = np.full((60, 40), 255, np.uint8)
            cv2.putText(image, char, (4, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, thickness)
            images.append(image)
            labels.append(char)
    model = make_svm(kernel="linear").fit(HOGExtractor().transform(images), labels)
    pipeline = PlateRecognizer(model, DetectionConfig(cascade_path=str(cascade)))
    plate = np.full((80, 160, 3), 255, np.uint8)
    cv2.putText(plate, "2Z", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    before = pipeline.recognize_crop(plate)
    registry = ModelRegistry(tmp_path / "Registration")
    registry.register("pipeline", pipeline.to_bundle(), {})
    bundle, _ = registry.load("pipeline", "v1", trusted=True)
    restored = PlateRecognizer.from_bundle(bundle, tmp_path / "relocated-assets")
    after = restored.recognize_crop(plate)
    assert len(before.segmentation.characters) == 2
    np.testing.assert_allclose(before.probabilities, after.probabilities)
    assert before.decoded == after.decoded
    assert restored.detector.config.cascade_path != str(cascade)
    result = restored.detector.detect(np.zeros((100, 200, 3), np.uint8))
    assert result.diagnostics["haar_enabled"]
    assert result.candidates == []
