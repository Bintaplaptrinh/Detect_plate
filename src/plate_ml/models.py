import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


def make_svm(c=1.0, kernel="rbf", seed=42, probability=True):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel=kernel, probability=probability,
                         class_weight="balanced", random_state=seed))


class ConfusionExperts(ClassifierMixin, BaseEstimator):
    """Classical, MoE-inspired conditional experts; no neural routing.

    A logistic gate learns group membership from training HOG. Each local SVM
    redistributes probability *within* its confusion group, preserving its mass.
    Global model and all experts are trained only on the training partition.
    This is supervised routing, not joint EM optimization of the original MoE.
    """

    def __init__(self, c=1.0, kernel="rbf", groups=("Z2", "A4", "I1", "3B8"), strength=0.5, seed=42):
        self.c = c
        self.kernel = kernel
        self.groups = groups
        self.strength = strength
        self.seed = seed

    def fit(self, X, y):
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        y = np.asarray(y)
        self.global_ = make_svm(self.c, self.kernel, self.seed).fit(X, y)
        self.classes_ = self.global_.classes_
        self.n_features_in_ = X.shape[1]
        used, self.experts_ = set(), []
        gate_y = np.full(len(y), "other", dtype=object)
        for index, group in enumerate(self.groups):
            labels = [c for c in self.classes_ if c in group]
            if used.intersection(labels):
                raise ValueError("Expert groups must be disjoint")
            used.update(labels)
            mask = np.isin(y, labels)
            if len(labels) < 2:
                continue
            expert = make_svm(self.c, self.kernel, self.seed + index + 1).fit(X[mask], y[mask])
            key = f"group_{index}"
            gate_y[mask] = key
            self.experts_.append((key, expert))
        self.gate_ = None
        self.single_gate_ = gate_y[0]
        if len(set(gate_y)) > 1:
            self.gate_ = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000,
                                      class_weight="balanced", random_state=self.seed)).fit(X, gate_y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "global_")
        result = self.global_.predict_proba(X)
        gates = self.gate_.predict_proba(X) if self.gate_ is not None else None
        for key, expert in self.experts_:
            columns = [list(self.classes_).index(label) for label in expert.classes_]
            mass = result[:, columns].sum(axis=1, keepdims=True)
            gate = gates[:, list(self.gate_.classes_).index(key), None] if gates is not None else np.ones((len(X), 1))
            blend = self.strength * gate
            result[:, columns] = (1 - blend) * result[:, columns] + blend * mass * expert.predict_proba(X)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
