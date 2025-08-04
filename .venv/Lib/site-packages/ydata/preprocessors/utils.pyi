from _typeshed import Incomplete
from collections.abc import Iterable
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from ydata.preprocessors.preprocess_methods import CategoricalEncoder, CategoricalImputer, DateTimeImputer, DateTimeTransformer, IdentityTransformer, IntegerTransformer

class TimeSeriesIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X): ...
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesEquidistance(BaseEstimator, TransformerMixin):
    mode: Incomplete
    def fit(self, X): ...
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesOutlierDetector(BaseEstimator, TransformerMixin):
    outlier_frac: Incomplete
    def __init__(self, *, outlier_frac: float = 0.01) -> None: ...
    model: Incomplete
    def fit(self, X): ...
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesDetrend(BaseEstimator, TransformerMixin):
    cols_to_detrend: Incomplete
    trends: Incomplete
    def __init__(self, cols_to_detrend) -> None: ...
    def fit(self, X): ...
    def fit_column(self, X): ...
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesMinMaxScaler(MinMaxScaler):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesIntegerTransformer(IntegerTransformer):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesIdentityTransformer(IdentityTransformer):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesCategoricalImputer(CategoricalImputer):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesCategoricalEncoder(CategoricalEncoder):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesDateTimeImputer(DateTimeImputer):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class TimeSeriesDateTimeTransformer(DateTimeTransformer):
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X): ...

class LambertWTransformer(BaseEstimator, TransformerMixin):
    """Applies the Lambert W transform.

    Note: Makes the tails of the distribution heavier. e.g. Approximate a gaussian to a returns distribution.
    """
    delta: Incomplete
    def __init__(self, delta: float = 0.6) -> None:
        """Inits the transformer with the parameter delta."""
    def fit(self, X, y: Incomplete | None = None): ...
    def transform(self, X):
        """Applies the Lambert W transform."""
    def inverse_transform(self, X):
        """Inverse transform back to the original series."""

class LogReturnsTransformer(BaseEstimator, TransformerMixin):
    """Calculates the log returns of a time series.

    Note: Can't handle negative values
    """
    def __init__(self) -> None: ...
    def fit(self, X, y: Incomplete | None = None): ...
    def transform(self, X):
        """Transform the series into log returns."""
    def inverse_transform(self, X, start: int = 0):
        """Given a starting point transforms the log returns into the original
        series."""

class ArcSinHTransformer(BaseEstimator, TransformerMixin):
    """Applies the Hiperbolic ArcSin transformation to a time series.

    Note: Can be used to transform negative prices. e.g. energy prices use case
    """
    offset: Incomplete
    scale: Incomplete
    def __init__(self, offset, scale) -> None:
        """Inits the transformer with the offset and scale."""
    def fit(self, X, y: Incomplete | None = None): ...
    def transform(self, X):
        """Applies the ArcSinH transform to the series."""
    def inverse_transform(self, X):
        """Inverse transform back to the original series."""

class AbsReturnsTransformer(BaseEstimator, TransformerMixin):
    """Calculates the absolute returns of a time series."""
    def __init__(self) -> None: ...
    def fit(self, X, y: Incomplete | None = None): ...
    def transform(self, X: DataFrame):
        """Transform into absolute returns."""
    def inverse_transform(self, X: DataFrame, start: float = 0):
        """Given a starting point transforms the returns into the original
        series."""

class RealToBinaryTransformer(BaseEstimator, TransformerMixin):
    """Transform a univariate series of real values into binary values.

    MinMax scaling is applied to the real values before converting to
    binary. Implemented from: The Market Generator, A. Kondratyev,
    Christian Schwarz
    """
    max: Incomplete
    min: Incomplete
    n_digits: Incomplete
    def __init__(self) -> None: ...
    def fit(self, X, y: Incomplete | None = None, n_binary_digits: int = 16):
        """Retrieves the min and max from a given series X."""
    def transform(self, X: DataFrame):
        """Transform real values into binary."""
    def inverse_transform(self, X: DataFrame):
        """Transforms binary values into real."""

class NDiffTransformer(BaseEstimator, TransformerMixin):
    """Calculates the n-th differences of a time series.

    Note: Requires a starting value for inverse transform
    """
    start: Incomplete
    def __init__(self) -> None: ...
    def fit(self, X: DataFrame, y: Incomplete | None = None): ...
    def transform(self, X: DataFrame, n_diff: int = 1):
        """Transform the series into n-th differences.

        Args:
            X (pandas.DataFrame): Data to transform.
            n_diff (int): Subtract the current value x(t) by x(t-n_diff).
        """
    def inverse_transform(self, X, start: Iterable = None):
        """Given a starting point transforms the differences into the original
        series.

        Args:
            start (Iterable): Starting values of original series.
        """
