import numpy as np
from constrained_linear_regression import ConstrainedLinearRegression
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression


def test_unconstrained():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    model = ConstrainedLinearRegression(nonnegative=False)
    model.fit(X, y)
    baseline = LinearRegression()
    baseline.fit(X, y)
    assert model.coef_.min() < 0
    assert np.allclose(baseline.coef_, model.coef_)
    assert np.isclose(baseline.intercept_, model.intercept_)


def test_positive():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    model = ConstrainedLinearRegression(nonnegative=True)
    model.fit(X, y)
    assert np.all(model.coef_ >= 0)
