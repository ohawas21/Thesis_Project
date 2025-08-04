from typing import Tuple

import numpy as np
import pyarrow
from dask.dataframe import DataFrame as ddDataFrame
from pandas import CategoricalDtype, StringDtype

from ydata.utils.type_inference import TypeConverter


def timeseries_split(
    dd: ddDataFrame, holdout_frac: float, is_sorted=True
) -> Tuple[ddDataFrame]:
    """Splits a timeseries DataFrame, with a temporal index, into train and
    holdout.

    Pass False in is_sorted to do pre-sorting of the index.
    """
    index = list(dd.index)
    if not is_sorted:
        index.sort()
    n_rows = len(index)
    split = int(n_rows * (1 - holdout_frac))
    train_idx = index[:split]
    holdout_idx = index[split:]
    return dd.loc[train_idx], dd.loc[holdout_idx]


def humanize_dtypes(schema) -> tuple:
    "Standardizes the data types to user-friendly representation."
    dtypes = {}
    dd_dtypes = {}
    for col, v in schema.items():
        if isinstance(v, CategoricalDtype):
            v = v.categories.dtype
        elif isinstance(v, StringDtype):
            v = str

        aux = TypeConverter.from_low(v)
        dtypes[col] = aux
        if aux in ["date", "datetime", "time"]:
            dd_dtypes[col] = v
        else:
            dd_dtypes[col] = aux
    return dtypes, dd_dtypes
