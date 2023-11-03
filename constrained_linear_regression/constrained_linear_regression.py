try:
    from sklearn.linear_model._base import LinearModel, _preprocess_data
except ImportError:
    from sklearn.linear_model.base import LinearModel, _preprocess_data

from .base import BaseConstrainedLinearRegression
from sklearn.base import RegressorMixin
import numpy as np

class ConstrainedLinearRegression(BaseConstrainedLinearRegression, RegressorMixin):
    def fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None):
        X, y, X_offset, y_offset, X_scale = self.preprocess(X, y)
        min_coef = min_coef if min_coef else np.repeat(-np.inf, X.shape[1])
        max_coef = max_coef if max_coef else np.repeat(np.inf, X.shape[1])
        beta = initial_beta if initial_beta else np.zeros(X.shape[1]).astype(float)

        if self.nonnegative:
            min_coef = np.clip(min_coef, 0, None)

        prev_beta = beta + 1
        hessian = self.calculate_hessian(X)
        loss_scale = len(y)

        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print('THE MODEL DID NOT CONVERGE')
                break
            
            step += 1
            prev_beta = beta.copy()

            for i in range(len(beta)):
                grad = self.calculate_gradient(X, beta, y)
                beta = self.update_beta(beta, i, grad, hessian, loss_scale, min_coef, max_coef)

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def calculate_hessian(self, X):
        hessian = np.dot(X.transpose(), X)
        if self.ridge:
            hessian += np.eye(X.shape[1]) * self.ridge
        return hessian

    def calculate_gradient(self, X, beta, y):
        grad = np.dot(np.dot(X, beta) - y, X)
        if self.ridge:
            grad += beta * self.ridge
        return grad

    def update_beta(self, beta, i, grad, hessian, loss_scale, min_coef, max_coef):
        prev_value = beta[i]
        new_value = beta[i] - grad[i] / hessian[i, i] * self.learning_rate
        if self.lasso:
            new_value = self.apply_lasso(beta, i, grad, hessian, loss_scale, prev_value, new_value)
        return np.clip(new_value, min_coef[i], max_coef[i])

    def apply_lasso(self, beta, i, grad, hessian, loss_scale, prev_value, new_value):
        new_value2 = (beta[i] - (grad[i] + np.sign(prev_value or new_value) * self.lasso * loss_scale)
        / hessian[i, i] * self.learning_rate)
        return 0 if new_value2 * new_value < 0 else new_value2