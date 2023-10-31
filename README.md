# constrained-linear-regression with Darts

This package also provides out-of-box compatibility with [Unit8's Darts](https://unit8co.github.io/darts/), a forecasting library. Darts supports a rich variety of models including but not limited to ARIMA, Prophet, Theta, and a host of others. 

One of the major features of `MultiConstrainedLinearRegression` when used with Darts, is the ability to set different constraints for coefficients for each custom horizon.

Here's an example using dummy data:

```Python
import numpy as np
import pandas as pd
from constrained_linear_regression import MultiConstrainedLinearRegression
from darts.models import RegressionModel
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler

# Create two pandas dataframes df_feature and df_target
date_rng = pd.date_range(start='1/1/2021', end='1/31/2021', freq='D')
df_feature = pd.DataFrame(date_rng, columns=['ds'])
df_feature['y'] = np.random.randint(0,100,size=(len(date_rng)))

df_target = pd.DataFrame(date_rng, columns=['ds'])
df_target['y'] = np.random.randint(0,100,size=(len(date_rng)))

# Load dataframes into time series
series_feature = TimeSeries.from_dataframe(df_feature, 'ds', 'y')
series_target = TimeSeries.from_dataframe(df_target, 'ds', 'y')

horizon = 14 # forecast horizon days
total_lags = 7 # past days to consider
custom_model = MultiConstrainedLinearRegression(fit_intercept=False, nonnegative=True)

model = RegressionModel(lags=1, output_chunk_length=horizon, lags_past_covariates=total_lags, model=custom_model, multi_models=True,
                       add_encoders={'transformer': Scaler()})
```
```Python
# Here we set different minimum and maximum constraints for coefficients for each horizon.
min_coef = np.zeros((horizon, total_lags))
max_coef = np.ones((horizon, total_lags))*np.inf

# Custom constraints
max_coef[:,0] = 0 # don't use target
max_coef[1:,14] = 0 
max_coef[2:,1] = 0 
...
max_coef[:,13] = 0 # ignore the last
```
```Python
model.fit(series_target, past_covariates=series_feature, horizon=horizon)

# Reset global variable
MultiConstrainedLinearRegression.global_horizon_count = 0

# Predict the next 14 days, using the past 7 days.
pred = model.predict(n=horizon, past_covariates=series_feature)

series_target[-(2*horizon):].plot(label='actual')
pred.plot(label='forecast')

# Output the learned coefficients and intercepts
for i, estimator in enumerate(model.model.estimators_):
    print(f"The coefficients for day {i+1} are {estimator.coef_}")
    print(f"The intercept for day {i+1} is {estimator.intercept_}")
```
In this example, the `MultiConstrainedLinearRegression` model is used with Darts' `RegressionModel` for sequence forecasts. We apply different constraints for each day within the forecast horizon, controlled by the `min_coef` and `max_coef` parameters. 

This gives us the flexibility to independently control the linearity of the regression model for different periods within the forecast horizon.
