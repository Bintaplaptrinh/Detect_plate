from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from plate_ml.config import TrainingConfig
from plate_ml.evaluation import evaluate_classifier, sequence_metrics, detection_metrics
from plate_ml.models import ConfusionExperts
from plate_ml.registry import ModelRegistry
from plate_ml.training import train_ocr


def learning_data():
    X, y = make_classification(n_samples=180, n_features=12, n_informative=9, n_redundant=0,
                               n_classes=3, class_sep=1.4, random_state=12)
    return X, np.asarray(["2ZB"[i] for i in y])


def test_experts_probabilities_and_mass_conservation():
    X, y = learning_data()
    model = ConfusionExperts(groups=("2Z",), seed=12).fit(X, y)
    base = model.global_.predict_proba(X)
    actual = model.predict_proba(X)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    b = list(model.classes_).index("B")
    np.testing.assert_allclose(actual[:, b], base[:, b])
    model.strength = 0
    np.testing.assert_allclose(model.predict_proba(X), base)


def test_training_early_stopping_and_selected_validation_model():
    X, y = learning_data()
    Xtr, Xval, ytr, yval = train_test_split(X, y, stratify=y, random_state=42)
    config = TrainingConfig(svm_c=(1,), svm_kernels=("linear",), ada_depths=(2,),
                            ada_learning_rates=(0.5,), ada_max_estimators=20,
                            patience=2, min_delta=1.0, expert_groups=("2Z",))
    saved = []
    result = train_ocr(Xtr, ytr, Xval, yval, config, checkpoint=lambda name, model, data: saved.append(name))
    assert len(next(iter(result.histories.values()))) <= 3
    assert len(result.models["adaboost"].estimators_) == 1
    assert set(result.models) == {"svm", "adaboost", "confusion_experts"}
    assert result.validation[result.champion]["macro_f1"] == max(m["macro_f1"] for m in result.validation.values())
    assert set(saved) == set(result.models)
    for name, model in result.models.items():
        assert evaluate_classifier(model, Xval, yval)["accuracy"] >= 0


def test_registry_versions_tampering_concurrency_and_path_guard(tmp_path):
    registry = ModelRegistry(tmp_path / "Registration")
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda i: registry.register("ocr", {"run": i}, {"metric": 0.9}), range(3)))
    assert {r["version"] for r in records} == {"v1", "v2", "v3"}
    assert len(registry.list_versions("ocr")) == 3
    with pytest.raises(ValueError):
        registry.load("ocr", "v1")
    bundle, _ = registry.load("ocr", "v1", trusted=True)
    assert "run" in bundle
    (registry.root / "Registry" / "ocr" / "v1" / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        registry.load("ocr", "v1", trusted=True)
    with pytest.raises(ValueError):
        registry.register("../escape", {}, {})
    with pytest.raises(ValueError):
        registry.load("ocr", "../../escape", trusted=True)


def test_evaluation_counts_failures_and_duplicate_detections():
    assert sequence_metrics(["ABC", "12"], ["ABC", ""]) == dict(exact_match=0.5, character_error_rate=0.4, samples=2)
    result = detection_metrics([[(0, 0, 10, 10), (0, 0, 10, 10)]], [[(0, 0, 10, 10)]])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
