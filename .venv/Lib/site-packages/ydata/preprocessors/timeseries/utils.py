from numpy import copy, exp, sign, sqrt, where
from pandas import DataFrame
from scipy.special import lambertw
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

from ydata.preprocessors.preprocess_methods import IdentityTransformer, IntegerTransformer


class TimeSeriesIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class TimeSeriesOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, *, outlier_frac: float = 0.01):
        super().__init__(novelty=True)
        self.outlier_frac = outlier_frac

    def fit(self, X):
        self.model = IsolationForest(contamination=self.outlier_frac)
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        outliers_map = self.model.predict(X)
        outliers_idx = where(outliers_map == -1)
        return X.drop(X.index[outliers_idx])

    def inverse_transform(self, X):
        return X


class TimeSeriesDetrend(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_detrend):
        self.cols_to_detrend = cols_to_detrend
        self.trends = {}

    def fit(self, X):
        for column in self.cols_to_detrend:
            self.trends[column] = self.fit_column(X[column].astype(float))
        return self

    def fit_column(self, X):
        trend = seasonal_decompose(
            X, model="addicative", extrapolate_trend="freq", period=1
        )
        return trend

    def transform(self, X, y=None):
        for column in self.cols_to_detrend:
            X[column] = X[column] - self.trends[column].trend
        return X

    def inverse_transform(self, X):
        for column in self.cols_to_detrend:
            X[column] = X[column] + self.trends[column].trend
        return X


class TimeSeriesIntegerTransformer(IntegerTransformer):
    def transform(self, X, y=None):
        X = DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        return X


class TimeSeriesIdentityTransformer(IdentityTransformer):
    def transform(self, X, y=None):
        X = DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        return X


class LambertWTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, delta=0.6):
        self.delta = delta
        super(BaseEstimator, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = copy(X)
        return (sign(X) * sqrt(lambertw(self.delta * X**2) / self.delta)).real

    def inverse_transform(self, X):
        X = copy(X)
        return X * exp(0.5 * self.delta * X**2)
