from collections.abc import Iterable

from numpy import (arcsinh, array, asarray, binary_repr, copy, diff, divide, empty, exp, log, max, min, nan_to_num,
                   power, reshape, roll, sign, sinh, sqrt, where)
from pandas import DataFrame, Series, to_datetime
from scipy.special import lambertw
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from ydata.preprocessors.preprocess_methods import (CategoricalEncoder, CategoricalImputer, DateTimeImputer,
                                                    DateTimeTransformer, IdentityTransformer, IntegerTransformer)


class TimeSeriesIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class TimeSeriesEquidistance(BaseEstimator, TransformerMixin):
    def fit(self, X):
        self.mode = (Series(X.index) - Series(X.index).shift(1)).mode().iloc[0]
        return self

    def transform(self, X, y=None):
        return X.resample(self.mode).asfreq()

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


class TimeSeriesMinMaxScaler(MinMaxScaler):
    def transform(self, X, y=None):
        X = DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = DataFrame(
            data=super().inverse_transform(X), columns=X.columns, index=X.index
        )
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


class TimeSeriesCategoricalImputer(CategoricalImputer):
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


class TimeSeriesCategoricalEncoder(CategoricalEncoder):
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


class TimeSeriesDateTimeImputer(DateTimeImputer):
    def transform(self, X, y=None):
        X = DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = DataFrame(
            data=super().inverse_transform(X), columns=X.columns, index=X.index
        )
        return X


class TimeSeriesDateTimeTransformer(DateTimeTransformer):
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
        for col_name in X.columns.to_list():
            X[col_name] = to_datetime(X[col_name])
        return X


class LambertWTransformer(BaseEstimator, TransformerMixin):
    """Applies the Lambert W transform.

    Note: Makes the tails of the distribution heavier. e.g. Approximate a gaussian to a returns distribution.
    """

    def __init__(self, delta=0.6):
        """Inits the transformer with the parameter delta."""
        self.delta = delta
        super(BaseEstimator, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Applies the Lambert W transform."""
        X = copy(X)
        return (sign(X) * sqrt(lambertw(self.delta * X**2) / self.delta)).real

    def inverse_transform(self, X):
        """Inverse transform back to the original series."""
        X = copy(X)
        return X * exp(0.5 * self.delta * X**2)


class LogReturnsTransformer(BaseEstimator, TransformerMixin):
    """Calculates the log returns of a time series.

    Note: Can't handle negative values
    """

    def __init__(self):
        super(BaseEstimator, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform the series into log returns."""
        X_ = copy(X)
        assert not any(
            X_ < 0), "LogReturns transform can't handle negative values."
        return nan_to_num(log(divide(X_, roll(X_, 1))), nan=0.0)

    def inverse_transform(self, X, start=0):
        """Given a starting point transforms the log returns into the original
        series."""
        X_ = copy(X)
        X_ = exp(X_)
        post_return = empty((X_.shape[0],))
        post_return[0] = start
        for i in range(1, X_.shape[0]):
            post_return[i] = post_return[i - 1] * X_[i]
        return reshape(post_return, (-1, 1))


class ArcSinHTransformer(BaseEstimator, TransformerMixin):
    """Applies the Hiperbolic ArcSin transformation to a time series.

    Note: Can be used to transform negative prices. e.g. energy prices use case
    """

    def __init__(self, offset, scale):
        """Inits the transformer with the offset and scale."""
        self.offset = offset
        self.scale = scale
        super(BaseEstimator, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Applies the ArcSinH transform to the series."""
        X = copy(X)
        return arcsinh((X - self.offset) / self.scale)

    def inverse_transform(self, X):
        """Inverse transform back to the original series."""
        X = copy(X)
        return self.offset + self.scale * sinh(X)


class AbsReturnsTransformer(BaseEstimator, TransformerMixin):
    """Calculates the absolute returns of a time series."""

    def __init__(self):
        super(BaseEstimator, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame):
        """Transform into absolute returns."""
        X_ = copy(X)
        return nan_to_num(divide(X_ - roll(X_, 1), roll(X_, 1)), nan=0.0)

    def inverse_transform(self, X: DataFrame, start: float = 0):
        """Given a starting point transforms the returns into the original
        series."""
        X_ = copy(X).flatten()
        post_return = empty((X_.shape[0],))
        post_return[0] = start
        for i in range(1, X_.shape[0]):
            post_return[i] = post_return[i - 1] + post_return[i - 1] * X_[i]
        return reshape(post_return, (-1, 1))


class RealToBinaryTransformer(BaseEstimator, TransformerMixin):
    """Transform a univariate series of real values into binary values.

    MinMax scaling is applied to the real values before converting to
    binary. Implemented from: The Market Generator, A. Kondratyev,
    Christian Schwarz
    """

    def __init__(self):
        super(BaseEstimator, self).__init__()
        self.max = None
        self.min = None
        self.n_digits = None

    def fit(self, X, y=None, n_binary_digits=16):
        """Retrieves the min and max from a given series X."""
        self.max = max(asarray(X))
        self.min = min(asarray(X))
        self.n_digits = n_binary_digits
        return self

    def transform(self, X: DataFrame):
        """Transform real values into binary."""
        X_ = copy(X)
        X_int = divide(
            (power(2, self.n_digits) - 1) *
            (X_ - self.min), (self.max - self.min)
        ).astype(int, copy=False)
        X_bin = empty((X_.shape[0], self.n_digits))

        for e in range(X_int.shape[0]):
            X_bin[e] = array(
                list(binary_repr(X_int[e][0]).zfill(self.n_digits))
            ).astype(int)
        return X_bin

    def inverse_transform(self, X: DataFrame):
        """Transforms binary values into real."""
        X_ = copy(X)
        X_real = empty(X_.shape[0])
        for e in range(X_.shape[0]):
            X_int = 0
            for m in range(self.n_digits):
                X_int = X_int + power(2, m) * X_[e][self.n_digits - 1 - m]
            X_real[e] = self.min + divide(
                X_int * (self.max - self.min), (power(2, self.n_digits) - 1)
            )
        return X_real.reshape(-1, 1)


class NDiffTransformer(BaseEstimator, TransformerMixin):
    """Calculates the n-th differences of a time series.

    Note: Requires a starting value for inverse transform
    """

    def __init__(self):
        super(BaseEstimator, self).__init__()
        self.start = None

    def fit(self, X: DataFrame, y=None):
        return self

    def transform(self, X: DataFrame, n_diff: int = 1):
        """Transform the series into n-th differences.

        Args:
            X (pandas.DataFrame): Data to transform.
            n_diff (int): Subtract the current value x(t) by x(t-n_diff).
        """
        X_ = copy(X)
        self.start = X_[0, :]
        return diff(X_, axis=0, n=n_diff)

    def inverse_transform(self, X, start: Iterable = None):
        """Given a starting point transforms the differences into the original
        series.

        Args:
            start (Iterable): Starting values of original series.
        """
        X_ = copy(X)
        post_diff = empty((X_.shape[0] + 1, X_.shape[1]))
        if self.start is None and start is None:
            start = [0 for n in range(X_.shape[1])]
        elif start is None:
            start = self.start

        post_diff[0, :] = start
        for i in range(1, X_.shape[0] + 1):
            post_diff[i, :] = post_diff[i - 1, :] + X_[i - 1, :]
        return post_diff
