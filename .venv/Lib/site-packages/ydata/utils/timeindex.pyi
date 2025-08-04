from pandas import DataFrame as DataFrame

def index_validation(X: DataFrame, col: str = None) -> DataFrame:
    """Auxiliary tool to validate a datasets given index.

    Args:
        col: str, with the name of the column. This is optional, as the user might provide a dataframe with the index already set.
    Returns: bool. 'True' in case the index set is valid and 'False' otherwise.
    """
def infer_frequency(X: DataFrame, partition_by: str | None = None):
    """Infers the frequency of non-continuous DatetimeIndex.

    Useful for time-series with missing values where the frequency is
    not automatically inferrable by pandas. Supports partition_by when
    data contains multiple entities.
    """
