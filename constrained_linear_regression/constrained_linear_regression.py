from .base import BaseConstrainedLinearRegression

import numpy as np


class ConstrainedLinearRegression(BaseConstrainedLinearRegression):
    """
    This class defines a version of Linear Regression that includes constraints on the coefficients. It extends the 
    BaseConstrainedLinearRegression class.

    Methods
    --------------
    fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None)
        This method is used to fit the ConstrainedLinearRegression model.

    Parameters
    --------------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Target vector relative to X.

    min_coef : array-like, shape (n_features,), optional
        Lower constraint for coefficients. Defaults to negative infinity for each coefficient.

    max_coef : array-like, shape (n_features,), optional
        Upper constraint for coefficients. Defaults to positive infinity for each coefficient.

    initial_beta : array-like, shape (n_features,), optional
        Initial coefficients to start the optimization. Defaults to zeros.

    Return
    --------------
    self : object
        Returns the instance itself.
    """
    def fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None):
        X, y, X_offset, y_offset, X_scale = self.preprocess(X, y)
        feature_count = X.shape[-1]
        min_coef_ = self._verify_coef(feature_count, min_coef, -np.inf).flatten()
        max_coef_ = self._verify_coef(feature_count, max_coef, np.inf).flatten()
        
        if self.nonnegative:
            min_coef_ = np.clip(min_coef_, 0, None)
        
        beta = self._verify_initial_beta(feature_count, initial_beta)

        prev_beta = beta + 1
        hessian = self._calculate_hessian(X)
        loss_scale = len(y)

        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print("THE MODEL DID NOT CONVERGE")
                break

            step += 1
            prev_beta = beta.copy()

            for i, _ in enumerate(beta):
                grad = self._calculate_gradient(X, beta, y)
                beta[i] = self._update_beta(
                    beta, i, grad, hessian, loss_scale, min_coef_, max_coef_
                )

        self._set_coef(beta)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def _calculate_hessian(self, X):
        hessian = np.dot(X.transpose(), X)
        if self.ridge:
            hessian += np.eye(X.shape[1]) * self.ridge
        return hessian

    def _calculate_gradient(self, X, beta, y):
        grad = np.dot(np.dot(X, beta) - y, X)
        if self.ridge:
            grad += beta * self.ridge
        return grad

    def _update_beta(self, beta, i, grad, hessian, loss_scale, min_coef, max_coef):
        prev_value = beta[i]
        new_value = beta[i] - grad[i] / hessian[i, i] * self.learning_rate
        if self.lasso:
            new_value = self._apply_lasso(
                beta, i, grad, hessian, loss_scale, prev_value, new_value
            )
        return np.clip(new_value, min_coef[i], max_coef[i])

    def _apply_lasso(self, beta, i, grad, hessian, loss_scale, prev_value, new_value):
        new_value2 = (
            beta[i]
            - (grad[i] + np.sign(prev_value or new_value) * self.lasso * loss_scale)
            / hessian[i, i]
            * self.learning_rate
        )
        return 0 if new_value2 * new_value < 0 else new_value2
