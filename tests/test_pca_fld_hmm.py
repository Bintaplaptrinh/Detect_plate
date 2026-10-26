from itertools import product
from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from plate_ml.models import make_pca_fld_svm
from plate_ml.hmm import VietnamesePlateHMM
from plate_ml.heuristics import PlateTemplate
from plate_ml.config import HOGConfig
from plate_ml.research_pipeline import ResearchRecognizer, RowProjectionSegmenter
from plate_ml.types import DetectionResult


def test_reduced_classifier_train_only_projection_and_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 12))
    y = np.repeat(list('ABC'), 20)
    X[:, :3] += np.eye(3)[np.repeat(np.arange(3), 20)] * 4
    model = clone(make_pca_fld_svm(pca_components=8)).fit(X, y)
    assert model.transform(X).shape == (60, 2)
    assert model.score(X, y) > .9
    before = deepcopy((model.scaler_.mean_, model.pca_.components_, model.fld_.scalings_))
    p = model.predict_proba(X[:5] + 10)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    for a, b in zip(before, (model.scaler_.mean_, model.pca_.components_, model.fld_.scalings_)):
        np.testing.assert_array_equal(a, b)
    path = tmp_path / 'candidate.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), model.predict_proba(X))


def test_reduced_classifier_caps_dimensions_and_rejects_invalid_classes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 5))
    model = make_pca_fld_svm(48, 20).fit(X, list('AAAAAABBBBBB'))
    assert model.pca_.n_components_ == 5
    assert model.output_features_ == 1
    with pytest.raises(ValueError, match='two classes'):
        model.fit(X, ['A'] * 12)


def test_viterbi_matches_exhaustive_sequence_probability():
    template = PlateTemplate('toy', ('12',) * 3, (3,))
    hmm = VietnamesePlateHMM((template,), max_log_penalty=10).fit(['112', '122', '222', '121'])
    classes = np.asarray(list('12'))
    p = np.asarray([[.6, .4], [.4, .6], [.7, .3]])
    initial = hmm.initial_ / hmm.initial_.sum()
    transition = hmm.transitions_ / hmm.transitions_.sum(axis=1, keepdims=True)
    def score(path):
        return initial[path[0]] * np.prod(p[np.arange(3), path]) * transition[path[0], path[1]] * transition[path[1], path[2]]
    best = max(product(range(2), repeat=3), key=score)
    assert hmm.decode(p, classes, (3,))['text'] == ''.join(classes[list(best)])


def test_hmm_template_correction_unknown_rows_and_probability_validation():
    classes = np.asarray(list('0123456789AB'))
    text = '30A12385'
    p = np.full((8, len(classes)), .001)
    for i, c in enumerate(text):
        p[i, list(classes).index(c)] = .9
    p[2, list(classes).index('4')] = .95
    hmm = VietnamesePlateHMM()
    assert hmm.decode(p, classes, (8,))['text'] == text
    assert hmm.decode(p, classes, (4, 4))['raw'] == '30412385'
    assert hmm.decode(p, classes, (2, 6))['text'] == '30412385'
    assert hmm.decode(np.empty((0, len(classes))), classes, ())['text'] == ''
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        hmm.decode(p, classes, (8,))


def test_hmm_penalty_fallback_and_fit_resets_counts():
    template = PlateTemplate('only_a', ('A',), (1,))
    hmm = VietnamesePlateHMM((template,))
    result = hmm.decode(np.array([[.001, .999]]), np.array(list('AB')), (1,))
    assert result['text'] == 'B' and result['template'] is None
    hmm.fit(['AA', 'AA'])
    before = hmm.transitions_.copy()
    hmm.fit(['AA'])
    np.testing.assert_array_equal(before, hmm.transitions_)
    assert hmm.training_sequences_ == 1


def test_real_arm_construction_supports_opencv_and_profiles_empty_pipeline():
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/run_pca_fld_hmm_comparison.py'))
    model = SimpleNamespace(classes_=np.array(list('12')))
    detector = SimpleNamespace(detect=lambda image: DetectionResult([], {}, {}))
    current = ResearchRecognizer(model, detector, RowProjectionSegmenter(), HOGConfig())
    arms = runner['build_arms'](current, model, VietnamesePlateHMM())
    assert len(arms) == 4
    for arm in arms.values():
        assert arm.extractor is not current.extractor
        crop = arm.recognize_crop(np.full((60, 180, 3), 255, np.uint8))
        assert crop['decoded']['text'] == ''
        assert all(value >= 0 for value in crop['stage_ms'].values())
        detection, predictions = arm.recognize(np.zeros((60, 180, 3), np.uint8))
        assert predictions == []
        assert detection.diagnostics['stage_ms']['detector'] >= 0
