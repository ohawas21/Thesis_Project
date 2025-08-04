import numpy as np
import pandas as pd
from _typeshed import Incomplete
from dask.dataframe.core import DataFrame as ddDataFrame

VALID_ENGINES: Incomplete
MB_PER_PART: int

def to_pandas(data: VALID_ENGINES) -> pd.DataFrame:
    """Converts data to pandas DataFrame."""
def to_numpy(data: VALID_ENGINES) -> np.ndarray:
    """Converts data to numpy array."""
def to_dask(data: VALID_ENGINES) -> ddDataFrame:
    """Converts data to dask array."""
