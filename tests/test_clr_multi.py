import numpy as np
from constrained_linear_regression.constrained_linear_regression import (
    ConstrainedLinearRegression,
)
from constrained_linear_regression.multi_constrained_linear_regression import (
    MultiConstrainedLinearRegression,
)
from sklearn.datasets import load_linnerud


def test_multi_constraint():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = ConstrainedLinearRegression(nonnegative=False)
    model = MultiConstrainedLinearRegression(nonnegative=False)

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        assert np.allclose(base_model.coef_, model.coef_), f"fails at {idx}th horizon."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_penalty_constraint():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    model = MultiConstrainedLinearRegression(nonnegative=False, penalty_rate=0.1)

    for idx in range(horizon):
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


if __name__ == "__main__":
    test_multi_constraint()
    test_multi_penalty_constraint()
