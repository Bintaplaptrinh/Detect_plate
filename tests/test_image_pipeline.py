from dataclasses import replace
import cv2
import numpy as np
import pytest
from plate_ml.config import SegmentationConfig, DetectionConfig
from plate_ml.detection import PlateDetector
from plate_ml.features import HOGExtractor, normalize_glyph
from plate_ml.imaging import intersection_over_union, rectify
from plate_ml.segmentation import CharacterSegmenter


def synthetic_plate(two_rows=False):
    image = np.full((140 if two_rows else 80, 300, 3), 255, np.uint8)
    rows = ("29A", "12345") if two_rows else ("29A12345",)
    for row, text in enumerate(rows):
        for index, char in enumerate(text):
            cv2.putText(image, char, (12 + index * 34, 52 + row * 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.circle(image, (285, 10), 3, (0, 0, 0), -1)
    cv2.circle(image, (155, image.shape[0] - 12), 2, (0, 0, 0), -1)
    return image


@pytest.mark.parametrize("two_rows", [False, True])
@pytest.mark.parametrize("method", ["cca", "projection", "cca_plus"])
def test_segmentation_preserves_narrow_glyphs_rejects_dots_orders_rows(two_rows, method):
    result = CharacterSegmenter(SegmentationConfig(method=method, threshold="otsu")).segment(synthetic_plate(two_rows))
    assert len(result.characters) == 8
    assert [c.row for c in result.characters] == ([0] * 3 + [1] * 5 if two_rows else [0] * 8)
    for row in set(c.row for c in result.characters):
        xs = [c.box[0] for c in result.characters if c.row == row]
        assert xs == sorted(xs)
    if method != "projection":
        assert any(not c["accepted"] for c in result.components)


def test_hog_polarity_blank_and_dimension():
    glyph = np.full((50, 35), 255, np.uint8)
    cv2.putText(glyph, "A", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 2)
    extractor = HOGExtractor()
    np.testing.assert_allclose(extractor.transform_one(glyph), extractor.transform_one(255 - glyph))
    assert extractor.transform([glyph]).shape == (1, 540)
    assert not normalize_glyph(np.zeros((20, 20), np.uint8)).any()
    assert extractor.transform([]).shape == (0, 540)


def test_detector_empty_scene_and_bad_model():
    result = PlateDetector().detect(np.zeros((160, 320, 3), np.uint8))
    assert result.candidates == []
    assert result.diagnostics["haar_enabled"] is False
    with pytest.raises(FileNotFoundError):
        PlateDetector(DetectionConfig(cascade_path="missing.xml"))


def test_rectification_and_iou():
    image = synthetic_plate()
    quad = np.float32([[0, 0], [299, 0], [299, 79], [0, 79]])
    warped = rectify(image, quad[[2, 0, 3, 1]])
    assert warped.shape == (79, 299, 3)
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert intersection_over_union((0, 0, 10, 10), (10, 0, 10, 10)) == 0
    portrait = np.zeros((120, 70, 3), np.uint8)
    assert rectify(portrait, np.float32([[0, 0], [69, 0], [69, 119], [0, 119]])).shape == (119, 69, 3)


def test_cca_does_not_split_horizontally_stretched_zero():
    image = np.full((60, 280, 3), 255, np.uint8)
    for i in range(6):
        cv2.ellipse(image, (25 + i * 44, 30), (16, 13), 0, 0, 360, (0, 0, 0), 2)
    result = CharacterSegmenter(SegmentationConfig(threshold="otsu")).segment(image)
    assert len(result.characters) == 6


def test_invalid_segmentation_config():
    with pytest.raises(ValueError):
        CharacterSegmenter(replace(SegmentationConfig(), method="typo"))
