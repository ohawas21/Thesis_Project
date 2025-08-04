"""
    Engines management
"""
import numpy as np
import pandas as pd

from dask.dataframe import from_pandas, from_array
from dask.dataframe.core import DataFrame as ddDataFrame

VALID_ENGINES = np.ndarray | pd.DataFrame | ddDataFrame
MB_PER_PART = 100  # MB per partition

def to_pandas(data: VALID_ENGINES) -> pd.DataFrame:
    "Converts data to pandas DataFrame."
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    elif isinstance(data, ddDataFrame):
        return data.compute()
    else:
        raise TypeError(
            f"Type {type(data)} is not supported by 'to_pandas' helper method."
        )


def to_numpy(data: VALID_ENGINES) -> np.ndarray:
    "Converts data to numpy array."
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, ddDataFrame):
        return data.compute().to_numpy()
    else:
        raise TypeError(
            f"Type {type(data)} is not supported by 'to_numpy' helper method."
        )


def to_dask(data: VALID_ENGINES) -> ddDataFrame:
    "Converts data to dask array."
    if isinstance(data, pd.DataFrame):
        memory_usage = data.memory_usage(index=True, deep=False).sum()
        npartitions = int(((memory_usage / 10e5) // MB_PER_PART) + 1)
        return from_pandas(data, npartitions=npartitions)
    elif isinstance(data, np.ndarray):
        return from_array(data, columns=[f'Col_{i}' for i in range(data.shape[1])])
    elif isinstance(data, ddDataFrame):
        return data
    else:
        raise TypeError(
            f"Type {type(data)} is not supported by 'to_dask' helper method."
        )
