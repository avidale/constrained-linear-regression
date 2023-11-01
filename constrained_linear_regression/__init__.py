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
            max_iter=10000,
            penalty_rate=0
    ):
        super().__init__(fit_intercept, normalize, copy_X, nonnegative, ridge, lasso, tol, learning_rate, max_iter)
        self.penalty_rate = penalty_rate

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
                if self.penalty_rate:
                    beta[i] = self.update_beta_penalty(beta, i, X, y, hessian, loss_scale, step/self.max_iter)
                else:
                    beta[i] = self.update_beta_clip(beta, i, X, y, hessian, loss_scale)

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        
        # Goes to the next multi-level model
        MultiConstrainedLinearRegression.global_horizon_count += 1
        return self
    
    def reset(self):
        MultiConstrainedLinearRegression.global_horizon_count = 0
        return f"global_horizon_count: {MultiConstrainedLinearRegression.global_horizon_count}"

    def update_beta_clip(self, beta, i, X, y, hessian, loss_scale):
        """
        This function updates the beta parameters with clipping to handle out-of-range beta values.

        :param beta: 1D numpy array, coefficients of the linear regression model
        :param i: int, index for the beta parameter to update
        :param X: 2D numpy array, input data for the linear model
        :param y: 1D numpy array, output data for the linear model
        :param hessian: 2D numpy array, Hessian matrix for the current parametrization of the model
        :param loss_scale: scalar, factor to rescale the loss function
        :return: new value for beta[i]

        The function first computes the gradient of the loss function. Then it performs one step of gradient descent 
        to estimate a new value for beta[i]. If specified, the function also applies lasso regularization.

        Finally, unlike the 'update_beta_penalty' function, this function enforces the constraints passively by simply 
        clipping the value of beta[i] so it remains within its specified range of values. This represents a simpler, 
        but sometimes less precise, way of imposing constraints on the model parameters.
        """
        grad = np.dot(np.dot(X, beta) - y, X)
        if self.ridge:
            grad += beta * self.ridge
        prev_value = beta[i]
        new_value = beta[i] - grad[i] / hessian[i,i] * self.learning_rate
        if self.lasso:
            new_value2 = beta[i] - (grad[i] + np.sign(prev_value or new_value) * self.lasso * loss_scale) / hessian[i,i] * self.learning_rate
            if new_value2 * new_value < 0:
                new_value = 0
            else:
                new_value = new_value2
        return np.clip(new_value, self.min_coef_[MultiConstrainedLinearRegression.global_horizon_count][i], self.max_coef_[MultiConstrainedLinearRegression.global_horizon_count][i])
    
    def update_beta_penalty(self, beta, i, X, y, hessian, loss_scale, progress):
        """
        This function updates the beta parameters with a penalty term if beta is out-of-range.

        :param beta: 1D numpy array, coefficients of the linear regression model
        :param i: int, index for the beta parameter to update
        :param X: 2D numpy array, input data for the linear model
        :param y: 1D numpy array, output data for the linear model
        :param hessian: 2D numpy array, Hessian matrix for the current parametrization of the model
        :param loss_scale: scalar, factor to rescale the loss function
        :param progress: scalar, progress of iteration to increase the penalty over iteration
        :return: new value for beta[i]

        The function first computes the gradient of the loss function, then adds an extra penalty term to it if beta[i] 
        is below/above the specified minimum/maximum coefficients. 

        Next, it performs one step of gradient descent to estimate a new value for beta[i], and finally, applies lasso 
        regularization if specified.

        Unlike other implementations that may clip the values of beta[i] to enforce constraints, this function incorporates
        the constraints into the optimization process itself by using penalty terms. In other words, it actively "discourages"
        the model from choosing out-of-range beta values instead of passively enforcing them by clipping.
        """
            
        grad = np.dot(np.dot(X, beta) - y, X) 
        grad += progress * self.penalty_rate * self.calc_distance_out_of_bounds(beta, i)
        
        if self.ridge:
            grad += beta * self.ridge
        prev_value = beta[i]
        new_value = beta[i] - grad[i] / hessian[i,i] * self.learning_rate
        if self.lasso:
            new_value2 = beta[i] - (grad[i] + np.sign(prev_value or new_value) * self.lasso * loss_scale) / hessian[i,i] * self.learning_rate
            if new_value2 * new_value < 0:
                new_value = 0
            else:
                new_value = new_value2
        return new_value

    def calc_distance_out_of_bounds(self, beta, i):
        min_bound = self.min_coef_[MultiConstrainedLinearRegression.global_horizon_count][i]
        max_bound = self.max_coef_[MultiConstrainedLinearRegression.global_horizon_count][i]
        if beta[i] < min_bound:
            return beta[i] - min_bound
        elif beta[i] > max_bound:
            return beta[i] - max_bound
        else:
            return 0

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
