from dataclasses import replace
import numpy as np
from .config import DetectionConfig
from .detection import PlateDetector
from .evaluation import detection_metrics
from .features import plate_features
from .imaging import read_image, intersection_over_union
from .models import make_svm
from .vnlp import verifier_dataset


def train_detector(train_scenes, validation_scenes, config=DetectionConfig(), seed=42):
    """Fit verifier, tune acceptance threshold on validation; return every trial."""
    X, y = verifier_dataset(train_scenes, seed=seed)
    verifier = make_svm(c=1.0, kernel="linear", seed=seed).fit(X, y)
    # Mine false proposals only from training scenes; never use val/test backgrounds.
    mining_detector = PlateDetector(replace(config, verifier_threshold=0.0), verifier)
    hard_negatives = []
    for scene in train_scenes:
        mined = mining_detector.detect(read_image(scene.path)).candidates
        hard_negatives.extend(plate_features(c.crop) for c in
                              [c for c in mined if intersection_over_union(c.box, scene.box) < 0.1][:4])
    if hard_negatives:
        X = np.vstack([X, hard_negatives])
        y = np.r_[y, np.zeros(len(hard_negatives), dtype=y.dtype)]
        verifier = make_svm(c=1.0, kernel="linear", seed=seed).fit(X, y)
    proposer = PlateDetector(replace(config, verifier_threshold=0.0), verifier)
    validation_candidates = [proposer.detect(read_image(s.path)).candidates for s in validation_scenes]
    trials = []
    for threshold in (0.3, 0.5, 0.7, 0.85, 0.95):
        predictions = [[c.box for c in items if c.score >= threshold] for items in validation_candidates]
        metrics = detection_metrics(predictions, [[s.box] for s in validation_scenes])
        trials.append(dict(threshold=threshold, training_examples=len(y), hard_negatives=len(hard_negatives), **metrics))
    best = max(trials, key=lambda t: (t["f1"], t["precision"]))
    return verifier, replace(config, verifier_threshold=best["threshold"]), trials


def evaluate_detector(detector, scenes):
    candidates = [detector.detect(read_image(s.path)).candidates for s in scenes]
    metrics = detection_metrics([[c.box for c in items] for items in candidates], [[s.box] for s in scenes])
    per_layout = {}
    for layout in sorted({s.layout for s in scenes}):
        indices = [i for i, s in enumerate(scenes) if s.layout == layout]
        per_layout[layout] = detection_metrics([[c.box for c in candidates[i]] for i in indices],
                                               [[scenes[i].box] for i in indices])
    top1 = detection_metrics([[items[0].box] if items else [] for items in candidates], [[s.box] for s in scenes])
    return dict(all_candidates=metrics, top1=top1, per_layout=per_layout)
