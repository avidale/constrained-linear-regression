import numpy as np
from constrained_linear_regression import ConstrainedLinearRegression
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression


def test_unconstrained():
    X, y = load_boston(return_X_y=True)
    model = ConstrainedLinearRegression(nonnegative=False)
    model.fit(X, y)
    baseline = LinearRegression()
    baseline.fit(X, y)
    assert np.allclose(baseline.coef_, model.coef_)
    assert np.isclose(baseline.intercept_, model.intercept_)


def test_positive():
    X, y = load_boston(return_X_y=True)
    model = ConstrainedLinearRegression(nonnegative=True)
    model.fit(X, y)
    assert np.all(model.coef_ >= 0)
