import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def make_svm(c=1.0, kernel="rbf", seed=42, probability=True):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel=kernel, probability=probability,
                         class_weight="balanced", random_state=seed))


class PCA_FLD_Classifier(ClassifierMixin, BaseEstimator):
    """Train-only StandardScaler -> PCA -> multiclass Fisher LDA -> SVC.

    LDA's SVD solver avoids explicitly inverting within-class scatter. PCA alone
    does not guarantee that scatter is nonsingular. Probabilities use the same
    SVC internal calibration as the current OCR classifier.
    """

    def __init__(self, pca_components=48, fld_components=None, c=3.0,
                 kernel="rbf", seed=42, probability=True):
        self.pca_components = pca_components
        self.fld_components = fld_components
        self.c, self.kernel, self.seed = c, kernel, seed
        self.probability = probability

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or np.any(counts < 2):
            raise ValueError("FLD requires at least two classes and two samples per class")
        if not isinstance(self.pca_components, int) or self.pca_components < 1:
            raise ValueError("pca_components must be a positive integer")
        n_pca = min(self.pca_components, X.shape[1], len(X) - len(classes))
        n_fld = min(len(classes) - 1, n_pca)
        if self.fld_components is not None:
            if not isinstance(self.fld_components, int) or self.fld_components < 1:
                raise ValueError("fld_components must be a positive integer")
            n_fld = min(n_fld, self.fld_components)
        self.scaler_ = StandardScaler().fit(X)
        scaled = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=n_pca, svd_solver="full").fit(scaled)
        reduced = self.pca_.transform(scaled)
        self.fld_ = LinearDiscriminantAnalysis(n_components=n_fld, solver="svd").fit(reduced, y)
        fisher = self.fld_.transform(reduced)
        if not fisher.shape[1]:
            raise ValueError("Training data has no discriminant directions")
        self.svm_ = SVC(C=self.c, kernel=self.kernel, probability=self.probability,
                        class_weight="balanced", random_state=self.seed).fit(fisher, y)
        self.classes_ = self.svm_.classes_
        self.output_features_ = fisher.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "svm_")
        X = validate_data(self, X, reset=False)
        return self.fld_.transform(self.pca_.transform(self.scaler_.transform(X)))

    def predict(self, X):
        return self.svm_.predict(self.transform(X))

    def predict_proba(self, X):
        return self.svm_.predict_proba(self.transform(X))

    def scatter_diagnostics(self, X, y):
        projected, y = self.transform(X), np.asarray(y)
        center = projected.mean(axis=0)
        within = between = 0.0
        for label in np.unique(y):
            group = projected[y == label]
            mean = group.mean(axis=0)
            within += float(np.square(group - mean).sum())
            between += len(group) * float(np.square(mean - center).sum())
        return dict(input_features=self.n_features_in_, pca_features=self.pca_.n_components_,
                    fld_features=self.output_features_, within_trace=within, between_trace=between,
                    separation_ratio=between / max(within, 1e-12),
                    pca_explained_variance_ratio_sum=float(self.pca_.explained_variance_ratio_.sum()))


def make_pca_fld_svm(pca_components=48, fld_components=None, c=3.0,
                     kernel="rbf", seed=42, probability=True):
    return PCA_FLD_Classifier(pca_components, fld_components, c, kernel, seed, probability)


class PCA_FLD_Gaussian(ClassifierMixin, BaseEstimator):
    """PCA/FLD observations with tied-covariance Gaussian HMM emissions.

    No SVC is created or fitted. Equal class priors make predict_proba the
    row-normalized emission likelihoods required by the sequence decoder.
    Means and pooled within-class covariance are estimated on train only.
    """

    def __init__(self, pca_components=48, fld_components=None, covariance_regularization=1e-6):
        self.pca_components = pca_components
        self.fld_components = fld_components
        self.covariance_regularization = covariance_regularization

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or np.any(counts < 2):
            raise ValueError("FLD requires at least two classes and two samples per class")
        if not isinstance(self.pca_components, int) or self.pca_components < 1:
            raise ValueError("pca_components must be a positive integer")
        if not np.isfinite(self.covariance_regularization) or self.covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be positive")
        n_pca = min(self.pca_components, X.shape[1], len(X) - len(counts))
        n_fld = min(len(counts) - 1, n_pca)
        if self.fld_components is not None:
            if not isinstance(self.fld_components, int) or self.fld_components < 1:
                raise ValueError("fld_components must be a positive integer")
            n_fld = min(n_fld, self.fld_components)
        self.scaler_ = StandardScaler().fit(X)
        scaled = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=n_pca, svd_solver="full").fit(scaled)
        reduced = self.pca_.transform(scaled)
        self.fld_ = LinearDiscriminantAnalysis(n_components=n_fld, solver="svd").fit(reduced, y)
        fisher = self.fld_.transform(reduced).astype(float)
        self.output_features_ = fisher.shape[1]
        if not self.output_features_:
            raise ValueError("Training data has no discriminant directions")
        self.means_ = np.asarray([fisher[y == c].mean(axis=0) for c in self.classes_])
        residual = fisher - self.means_[np.searchsorted(self.classes_, y)]
        covariance = residual.T @ residual / (len(y) - len(counts))
        scale = max(float(np.trace(covariance) / self.output_features_), np.finfo(float).eps)
        self.covariance_ = covariance + self.covariance_regularization * scale * np.eye(self.output_features_)
        chol = np.linalg.cholesky(self.covariance_)
        self.whitener_ = solve_triangular(chol, np.eye(self.output_features_), lower=True).T
        self.whitened_means_ = self.means_ @ self.whitener_
        self.log_normalizer_ = float(-.5 * (self.output_features_ * np.log(2*np.pi) + 2*np.log(np.diag(chol)).sum()))
        return self

    def transform(self, X):
        check_is_fitted(self, 'whitener_')
        X = validate_data(self, X, reset=False)
        return self.fld_.transform(self.pca_.transform(self.scaler_.transform(X)))

    def log_likelihood(self, X):
        whitened = self.transform(X).astype(float) @ self.whitener_
        distances = (np.square(whitened).sum(axis=1, keepdims=True)
                     + np.square(self.whitened_means_).sum(axis=1)
                     - 2 * whitened @ self.whitened_means_.T)
        return self.log_normalizer_ - .5 * np.maximum(distances, 0.)

    def predict_proba(self, X):
        return softmax(self.log_likelihood(X), axis=1)

    def predict(self, X):
        return self.classes_[self.log_likelihood(X).argmax(axis=1)]

    scatter_diagnostics = PCA_FLD_Classifier.scatter_diagnostics


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
