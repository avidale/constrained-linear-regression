try:
    from sklearn.linear_model._base import LinearModel, _preprocess_data
except ImportError:
    from sklearn.linear_model.base import LinearModel, _preprocess_data
import abc
from sklearn.utils import check_X_y

class BaseConstrainedLinearRegression(LinearModel):
    """
    Base class for ConstrainedLinearRegression,
    MatrixConstrainedLinearRegression and 
    MultiConstrainedLinearRegression.
    """
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

    def preprocess(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        return _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X)

    @abc.abstractmethod
    def fit(self, X, y):
        pass