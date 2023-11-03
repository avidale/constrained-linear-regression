import numpy as np
from constrained_linear_regression.constrained_linear_regression import ConstrainedLinearRegression
from constrained_linear_regression.multi_constrained_linear_regression import MultiConstrainedLinearRegression
from sklearn.datasets import load_linnerud
import pytest

@pytest.mark.xfail(raises=AssertionError)
def test_constraint_min_coef():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 4)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    model = ConstrainedLinearRegression(nonnegative=False)
    model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function

@pytest.mark.xfail(raises=AssertionError)
def test_constraint_max_coef():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 7)) * 2

    model = ConstrainedLinearRegression(nonnegative=False)
    model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function

@pytest.mark.xfail(raises=AssertionError)
def test_multi_constraint_min_coef():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 4)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    model = MultiConstrainedLinearRegression(nonnegative=False)
    model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function

@pytest.mark.xfail(raises=AssertionError)
def test_multi_constraint_max_coef():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 7)) * 2

    model = MultiConstrainedLinearRegression(nonnegative=False)
    model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function
