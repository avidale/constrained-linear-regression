# constrained-linear-regression
[![PyPI version](https://badge.fury.io/py/constrained-linear-regression.svg)](https://badge.fury.io/py/constrained-linear-regression)

This is a Python implementation of constrained linear regression in scikit-learn style. 
The current version supports upper and lower bound for each slope coefficient.

It was developed after this question https://stackoverflow.com/questions/50410037

Installation:
```pip install constrained-linear-regression```

You can use this model, for example, if you want all coefficients to be non-negative:

```Python
from constrained_linear_regression import ConstrainedLinearRegression
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
X, y = load_boston(return_X_y=True)
model = ConstrainedLinearRegression(nonnegative=True)
model.fit(X, y)
print(model.intercept_)
print(model.coef_)
```
The output will be like
```commandline
-36.99292986145538
[0.         0.05286515 0.         4.12512386 0.         8.04017956
 0.         0.         0.         0.         0.         0.02273805
 0.        ]
```
You can also impose arbitrary bounds for any coefficients you choose 
```Python
model = ConstrainedLinearRegression()
min_coef = np.repeat(-np.inf, X.shape[1])
min_coef[0] = 0
min_coef[4] = -1
max_coef = np.repeat(4, X.shape[1])
max_coef[3] = 2
model.fit(X, y, max_coef=max_coef, min_coef=min_coef)
print(model.intercept_)
print(model.coef_)
```
The output will be 
```commandline
24.060175576410515
[ 0.          0.04504673 -0.0354073   2.         -1.          4.
 -0.01343263 -1.17231216  0.2183103  -0.01375266 -0.7747823   0.01122374
 -0.56678676]
```

You can also set coefficients `lasso` and `ridge` if you want to apply the 
corresponding penalties. For `lasso`, however, the output might not be exactly 
equal to the result of `sklearn.linear_model.Lasso` due to the difference
in the optimization algorithm.
