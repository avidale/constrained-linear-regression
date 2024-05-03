try:
    from sklearn.linear_model._base import LinearModel, _preprocess_data
except ImportError:
    from sklearn.linear_model.base import LinearModel, _preprocess_data
import abc
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y


class BaseConstrainedLinearRegression(LinearModel, RegressorMixin):
    def __init__(
        self,
        fit_intercept=True,
        copy_X=True,
        nonnegative=False,
        ridge=0,
        lasso=0,
        tol=1e-15,
        learning_rate=1.0,
        max_iter=10000,
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.ridge = ridge
        self.lasso = lasso
        self.tol = tol
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def preprocess(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            y_numeric=True,
            multi_output=False,
        )
        return _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
        )

    def _verify_coef(self, feature_count, coef, value, idx=0):
        if coef is not None:
            coef_ = coef
            assert (
                coef_.shape[-1] == feature_count
            ), "Incorrect shape for coef_, the second dimension must match feature_count"
        else:
            coef_ = np.ones((idx + 1, feature_count)) * value
        return coef_

    def _verify_initial_beta(self, feature_count, initial_beta):
        if initial_beta is not None:
            beta = initial_beta
            assert beta.shape == (feature_count,), "Incorrect shape for initial_beta"
        else:
            beta = np.zeros(feature_count).astype(float)
        return beta

    def _set_coef(self, beta):
        self.coef_ = beta

    @abc.abstractmethod
    def fit(self, X, y):
        pass
