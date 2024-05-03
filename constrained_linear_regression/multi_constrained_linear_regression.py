import numpy as np
from .constrained_linear_regression import ConstrainedLinearRegression


class MultiConstrainedLinearRegression(ConstrainedLinearRegression):
    """
    This class implementation extends the ConstrainedLinearRegression class to handle multiple constraints.

    Attributes
    ----------
    global_horizon_count : int
        Static/class-level attribute used to keep track of the number of constraints across multiple levels/horizons.

    Methods
    --------
    fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None)
        Fit MultiConstrainedLinearRegression model to the given data.

    Parameters
    ----------
    X : array-like
        The input samples.

    y : array-like
        The target values.

    min_coef : array-like, default=None
        The minimum constraint for the coefficient. It's optional and default value is None

    max_coef : array-like
        The maximum constraint for the coefficient. It's optional and default value is None

    initial_beta : array-like
        Initial coefficients for starting the optimization process. It's optional and default value is None

    fit_intercept : bool, default=True
        Option to fit intercept.

    normalize : bool, default=False
        Option to normalize the data.

    copy_X : bool, default=True
        Option to copy X.

    nonnegative : bool, default=False
        Option to make sure coefficients are non-negative.

    ridge : int, default=0
        Regularization factor for ridge regression.

    lasso : int, default=0
        Regularization factor for lasso regression.

    tol : float, default=1e-15
        Tolerance for the stopping criterion.

    learning_rate : float, default=1.0
        Learning rate for the update rule.

    max_iter : int, default=10000
        Maximum number of iterations for the algorithm.

    penalty_rate : float, default=0
        Penalty rate applied to the coefficients.

    Return
    -------
    self : object
        Returns the instance itself.
    """

    global_horizon_count = 0  # Class attribute shared by all instances

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
        penalty_rate=0,
    ):
        super().__init__(
            fit_intercept,
            normalize,
            copy_X,
            nonnegative,
            ridge,
            lasso,
            tol,
            learning_rate,
            max_iter,
        )
        self.penalty_rate = penalty_rate

    def fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None):
        X, y, X_offset, y_offset, X_scale = self.preprocess(X, y)
        feature_count = X.shape[-1]
        min_coef_ = self._verify_coef(
            feature_count,
            min_coef,
            -np.inf,
            MultiConstrainedLinearRegression.global_horizon_count,
        )
        max_coef_ = self._verify_coef(
            feature_count,
            max_coef,
            np.inf,
            MultiConstrainedLinearRegression.global_horizon_count,
        )

        beta = self._verify_initial_beta(feature_count, initial_beta)

        if self.nonnegative:
            min_coef_ = np.clip(min_coef_, 0, None)

        prev_beta = beta + 1
        hessian = self._calculate_hessian(X)
        loss_scale = len(y)

        # Custom fit implementation starts from here
        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print("THE MODEL DID NOT CONVERGE")
                break

            step += 1
            prev_beta = beta.copy()

            for i, _ in enumerate(beta):
                grad = self._calculate_gradient(X, beta, y)
                if self.penalty_rate:
                    progress = step / self.max_iter
                    grad += (
                        progress
                        * self.penalty_rate
                        * self._calc_distance_out_of_bounds(
                            beta, i, min_coef_, max_coef_
                        )
                    )

                beta[i] = self._update_beta(
                    beta,
                    i,
                    grad,
                    hessian,
                    loss_scale,
                    min_coef[MultiConstrainedLinearRegression.global_horizon_count],
                    max_coef[MultiConstrainedLinearRegression.global_horizon_count],
                )

        self._set_coef(beta)
        self._set_intercept(X_offset, y_offset, X_scale)

        # Update horizon_count for the next model
        MultiConstrainedLinearRegression.global_horizon_count += 1

        return self

    def _calc_distance_out_of_bounds(self, beta, i, min_coef_, max_coef_):
        min_bound = min_coef_[MultiConstrainedLinearRegression.global_horizon_count][i]
        max_bound = max_coef_[MultiConstrainedLinearRegression.global_horizon_count][i]
        if beta[i] < min_bound:
            return beta[i] - min_bound
        elif beta[i] > max_bound:
            return beta[i] - max_bound
        else:
            return 0

    def reset(self):
        MultiConstrainedLinearRegression.global_horizon_count = 0
        return f"horizon_count: {MultiConstrainedLinearRegression.global_horizon_count}"
