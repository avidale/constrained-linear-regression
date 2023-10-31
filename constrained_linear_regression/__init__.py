try:
    from sklearn.linear_model._base import LinearModel, _preprocess_data
except ImportError:
    from sklearn.linear_model.base import LinearModel, _preprocess_data
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np


class ConstrainedLinearRegression(LinearModel, RegressorMixin):
    def __init__(
            self,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            nonnegative=False,
            ridge=0,
            lasso=0,
            tol=1e-15,
            learning_rate=1.0,
            max_iter=10000
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.ridge = ridge
        self.lasso = lasso
        self.tol = tol
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
        )
        self.min_coef_ = min_coef if min_coef is not None else np.repeat(-np.inf, X.shape[1])
        self.max_coef_ = max_coef if max_coef is not None else np.repeat(np.inf, X.shape[1])
        if self.nonnegative:
            self.min_coef_ = np.clip(self.min_coef_, 0, None)

        if initial_beta is not None:
            # providing initial_beta may be useful,
            # if initial solution does not respect the constraints.
            beta = initial_beta
        else:
            beta = np.zeros(X.shape[1]).astype(float)

        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        if self.ridge:
            hessian += np.eye(X.shape[1]) * self.ridge

        loss_scale = len(y)
        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print('THE MODEL DID NOT CONVERGE')
                break
            step += 1
            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X, beta) - y, X)
                if self.ridge:
                    grad += beta * self.ridge
                prev_value = beta[i]
                new_value = beta[i] - grad[i] / hessian[i,i] * self.learning_rate
                if self.lasso:
                    #
                    new_value2 = beta[i] - (grad[i] + np.sign(prev_value or new_value) * self.lasso * loss_scale) / hessian[i,i] * self.learning_rate
                    if new_value2 * new_value < 0:
                        new_value = 0
                    else:
                        new_value = new_value2
                beta[i] = np.clip(new_value, self.min_coef_[i], self.max_coef_[i])

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class MultiConstrainedLinearRegression(ConstrainedLinearRegression):
    """
    This class implementation extends the ConstrainedLinearRegression class to handle multiple constraints.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether or not to calculate the intercept for this model.

    normalize : bool, default=False
        This parameter is ignored when fit_intercept is set to False. 
        If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    nonnegative : bool, default=False
        When set to True, forces the coefficients to be positive.

    ridge : float, default=0
        When set to a float value, adds RIDGE regularization to the model, with the supplied alpha.

    lasso : float, default=0
        When set to a float value, adds LASSO regularization to the model, with the supplied alpha.

    tol : float, default=1e-15
        The tolerance for the optimization.

    learning_rate : float, default=1.0
        The learning rate for the optimization.

    max_iter : int, default=10000
        The maximum number of iterations for the optimization algorithm.

    Methods
    --------
    fit(X, y, horizon, min_coef=None, max_coef=None, initial_beta=None):
        Fits the model according to the given training data.
    """
    global_horizon_count = 0
    
    def __init__(
            self,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            nonnegative=False,
            ridge=0,
            lasso=0,
            tol=1e-15,
            learning_rate=1.0,
            max_iter=10000
    ):
        super().__init__(fit_intercept, normalize, copy_X, nonnegative, ridge, lasso, tol, learning_rate, max_iter)

    def fit(self, X, y, horizon, min_coef=None, max_coef=None, initial_beta=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
        )
        self.min_coef_ = min_coef if min_coef is not None else np.ones((horizon, X.shape[1]))* -np.inf
        self.max_coef_ = max_coef if max_coef is not None else np.ones((horizon, X.shape[1]))* np.inf
        if self.nonnegative:
            self.min_coef_ = np.clip(self.min_coef_, 0, None)

        if initial_beta is not None:
            # providing initial_beta may be useful,
            # if initial solution does not respect the constraints.
            beta = initial_beta
        else:
            beta = np.zeros(X.shape[1]).astype(float)

        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        if self.ridge:
            hessian += np.eye(X.shape[1]) * self.ridge

        loss_scale = len(y)
        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print('THE MODEL DID NOT CONVERGE')
                break
            step += 1
            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X, beta) - y, X)
                if self.ridge:
                    grad += beta * self.ridge
                prev_value = beta[i]
                new_value = beta[i] - grad[i] / hessian[i,i] * self.learning_rate
                if self.lasso:
                    #
                    new_value2 = beta[i] - (grad[i] + np.sign(prev_value or new_value) * self.lasso * loss_scale) / hessian[i,i] * self.learning_rate
                    if new_value2 * new_value < 0:
                        new_value = 0
                    else:
                        new_value = new_value2
                beta[i] = np.clip(new_value, self.min_coef_[MultiConstrainedLinearRegression.global_horizon_count][i], self.max_coef_[MultiConstrainedLinearRegression.global_horizon_count][i])

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        MultiConstrainedLinearRegression.global_horizon_count += 1
        return self


class MatrixConstrainedLinearRegression(LinearModel, RegressorMixin):
    """
    We want to fit a linear model $\\hat{Y}=\\alpha + X\\beta$
    with a linear constrant on $\\beta$: $A\\beta \\leq B$,
    where $A$ is a matrix, $B$ is a vector, and the vector inequality is element-wize.
    We do it by coordinate descent, recalculating constraints for every coefficient on each step
    """
    def __init__(self, A, B, fit_intercept=True, normalize=False, copy_X=True, tol=1e-15, lr=1.0):
        self.A = A
        self.B = B
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.lr = lr

    def fit(self, X, y, initial_beta=None):
        X, y = check_X_y(
            X, y,
            accept_sparse=['csr', 'csc', 'coo'],
            y_numeric=True, multi_output=False
        )
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X
        )
        if initial_beta is not None:
            # providing initial_beta may be useful,
            # if initial solution does not respect the constraints.
            beta = initial_beta
        else:
            beta = np.zeros(X.shape[1]).astype(float)
        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        while not (np.abs(prev_beta - beta)<self.tol).all():
            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X, beta) - y, X)
                max_coef = np.inf
                min_coef = -np.inf
                for a_row, b_value in zip(self.A, self.B):
                    if a_row[i] == 0:
                        continue
                    zero_out = beta.copy()
                    zero_out[i] = 0
                    bound = (b_value - np.dot(zero_out, a_row)) / a_row[i]
                    if a_row[i] > 0:
                        max_coef = np.minimum(max_coef, bound)
                    elif a_row[i] < 0:
                        min_coef = np.maximum(min_coef, bound)
                assert min_coef <= max_coef, "the constraints are inconsistent"
                beta[i] = np.minimum(
                    max_coef,
                    np.maximum(
                        min_coef,
                        beta[i] - (grad[i] / hessian[i,i]) * self.lr
                    )
                )
        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self
