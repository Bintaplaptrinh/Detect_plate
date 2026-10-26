import joblib
import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.base import clone
from sklearn.svm import SVC

from plate_ml.models import PCA_FLD_Gaussian


def training():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 12))
    y = np.repeat(list('ABC'), 30)
    X[:, :3] += np.eye(3)[np.repeat(np.arange(3), 30)] * 4
    return X, y


def test_gaussian_emissions_match_density_without_ever_fitting_svm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('The no-SVM observation model must never fit SVC')
    monkeypatch.setattr(SVC, 'fit', forbidden)
    X, y = training()
    model = clone(PCA_FLD_Gaussian(pca_components=8)).fit(X, y)
    z = model.transform(X[:8])
    expected = np.column_stack([multivariate_normal.logpdf(z, mean=m, cov=model.covariance_) for m in model.means_])
    np.testing.assert_allclose(model.log_likelihood(X[:8]), expected, atol=1e-10)
    assert model.output_features_ == 2
    assert model.score(X, y) > .95
    assert not hasattr(model, 'svm_')
    assert not any(isinstance(v, SVC) for v in vars(model).values())


def test_gaussian_train_only_roundtrip_and_finite_extreme_observations(tmp_path):
    X, y = training()
    model = PCA_FLD_Gaussian(8).fit(X, y)
    means = model.means_.copy()
    p = model.predict_proba(X[:4] * 1e5)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(means, model.means_)
    path = tmp_path / 'no_svm.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), model.predict_proba(X))


def test_gaussian_rejects_invalid_covariance_regularization():
    X, y = training()
    with pytest.raises(ValueError, match='regularization'):
        PCA_FLD_Gaussian(covariance_regularization=0).fit(X, y)
