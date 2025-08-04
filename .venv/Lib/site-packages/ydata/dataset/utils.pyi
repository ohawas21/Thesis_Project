from dask.dataframe import DataFrame as ddDataFrame

def timeseries_split(dd: ddDataFrame, holdout_frac: float, is_sorted: bool = True) -> tuple[ddDataFrame]:
    """Splits a timeseries DataFrame, with a temporal index, into train and
    holdout.

    Pass False in is_sorted to do pre-sorting of the index.
    """
def humanize_dtypes(schema) -> tuple:
    """Standardizes the data types to user-friendly representation."""
