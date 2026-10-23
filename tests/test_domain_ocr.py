from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plate_ml.config import HOGConfig, SegmentationConfig
from plate_ml.domain_ocr import (assert_group_disjoint, extract_vnlp_glyphs,
                                 remove_scene_duplicates, glyph_metrics)
from plate_ml.features import HOGExtractor
from plate_ml.segmentation import CharacterSegmenter
from plate_ml.vnlp import Scene


def scene_file(tmp_path, text='29A12345', rows=('29A12345',), name='plate.png'):
    image = np.full((140 if len(rows) == 2 else 80, 300, 3), 255, np.uint8)
    for row, value in enumerate(rows):
        for index, char in enumerate(value):
            cv2.putText(image, char, (12 + index * 34, 52 + row * 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    path = tmp_path / name
    cv2.imencode('.png', image)[1].tofile(path)
    scene = Scene(str(path), text, (0, 0, image.shape[1], image.shape[0]),
                  'two_rows' if len(rows) == 2 else 'one_row', text)
    return scene, image


@pytest.mark.parametrize('rows', [('29A12345',), ('29A', '12345')])
def test_training_features_equal_inference_and_labels_follow_reading_order(tmp_path, rows):
    scene, image = scene_file(tmp_path, rows=rows)
    config = SegmentationConfig(method='cca', threshold='otsu')
    data = extract_vnlp_glyphs([scene], config)
    assert data.y.tolist() == list(scene.text)
    assert data.X.shape == (8, 540)
    inference = CharacterSegmenter(config).segment(image)
    np.testing.assert_array_equal(data.X, HOGExtractor(HOGConfig()).transform(c.image for c in inference.characters))
    assert [r['row'] for r in data.records] == ([0] * 8 if len(rows) == 1 else [0] * 3 + [1] * 5)


def test_reject_whole_plate_for_count_or_wrong_row_alignment(tmp_path):
    mismatch, _ = scene_file(tmp_path, text='29A123456')
    # Same total count, but incompatible 2+6 alignment for all frozen templates.
    wrong_rows, _ = scene_file(tmp_path, rows=('29', 'A12345'), name='wrong_rows.png')
    data = extract_vnlp_glyphs([mismatch, wrong_rows], SegmentationConfig(method='cca', threshold='otsu'))
    assert data.X.shape == (0, 540)
    assert [p['reason'] for p in data.plates] == ['count_mismatch', 'row_or_format_mismatch']


def test_plate_identity_cannot_cross_splits(tmp_path):
    scene, _ = scene_file(tmp_path)
    with pytest.raises(ValueError, match='identity leakage'):
        assert_group_disjoint(dict(train=[scene], validation=[scene], test=[]))


def test_duplicate_pixels_with_conflicting_labels_fail(tmp_path):
    scene, _ = scene_file(tmp_path)
    conflict = Scene(scene.path, '30A12345', scene.box, scene.layout, '30A12345')
    with pytest.raises(ValueError, match='conflicting text'):
        remove_scene_duplicates(dict(train=[conflict], validation=[scene], test=[]))


def test_unseen_test_class_is_scored_as_error():
    model = SimpleNamespace(classes_=np.array(['0', '1']),
                            predict_proba=lambda X: np.tile([1.0, 0.0], (len(X), 1)))
    data = SimpleNamespace(X=np.zeros((2, 540)), y=np.array(['0', 'A']))
    metrics = glyph_metrics(model, data)
    assert metrics['accuracy'] == 0.5
    assert metrics['unseen_truth_classes'] == ['A']
    assert metrics['macro_f1'] == pytest.approx(1 / 3)
