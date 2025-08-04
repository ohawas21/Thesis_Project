from typing import Optional

from numpy import datetime64
from pandas import CategoricalIndex, DataFrame, DatetimeIndex, Index, MultiIndex, RangeIndex, TimedeltaIndex

from ydata.utils.formats import most_frequent

_UNSUPPORTED_INDEX = [
    RangeIndex,
    CategoricalIndex,
    Index,
    MultiIndex,
]


def index_validation(X: DataFrame, col: str = None) -> DataFrame:
    """Auxiliary tool to validate a datasets given index.

    Args:
        col: str, with the name of the column. This is optional, as the user might provide a dataframe with the index already set.
    Returns: bool. 'True' in case the index set is valid and 'False' otherwise.
    """
    if col is not None and type(X.index) is RangeIndex:
        dtype = X[col].dtypes.type
        if dtype == datetime64:
            index = DatetimeIndex(X[col])
        elif dtype == int:
            try:
                index = TimedeltaIndex(X[col], freq=None)
            except BaseException:
                raise Exception(
                    f"Please provide a valid integer column. Currently only {TimedeltaIndex} are supported."
                )
        else:
            raise Exception(
                "Please provide a valid column type as index. Currently only datetime and integers are supported."
            )
        X.index = index
        X = X.drop(col, axis=1)
    elif col is None:
        index_type = type(X.index)
        if index_type in _UNSUPPORTED_INDEX:
            raise Exception(
                f"Please provide an index column or set a valid index. {index_type.__name__} is not supported."
            )
    return X


def infer_frequency(X: DataFrame, partition_by: Optional[str] = None):
    """Infers the frequency of non-continuous DatetimeIndex.

    Useful for time-series with missing values where the frequency is
    not automatically inferrable by pandas. Supports partition_by when
    data contains multiple entities.
    """
    freqs = []
    if partition_by:
        for _, data in X.groupby(partition_by):  # for each partition
            freqs.append(
                data.reset_index()[X.index.name].diff().value_counts().idxmax()
            )
    else:
        freqs.append(X.reset_index()[
                     X.index.name].diff().value_counts().idxmax())
    return most_frequent(freqs)
