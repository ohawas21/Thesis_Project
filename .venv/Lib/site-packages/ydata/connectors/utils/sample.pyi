from dask import dataframe as dddataframe

def sample_fraction(df, sample: float | int, total_rows: int = None):
    """Sample either deterministic number of rows (exact, slow) or percentage
    of total (approximate, fast).

    Dask Dataframes API requires fraction sampling, so we convert into percentage
    of total if exact number of rows are requested.

    Examples
        >>> obs_connector.sample_fraction(df=None, sample=0.01)
        0.01
        >>> obs_connector.sample_fraction(df=None, sample=10, nrows=20)
        0.5

    Args:
        df: original dataframe
        sample (Union[float, int]): exact number of rows or percentage of total
        nrows (int, optional): number of rows if already calculated.

    Returns:
        calc_sample (float): applicable percentage to sample from dataset.
    """
def nsample(df: dddataframe, sample_size: int) -> dddataframe:
    """Obtain a contiguous sample from a Dask DataFrame with sample_size
    records.

    Args:
        df: original dataframe
        sample_size (int): requested number of rows for the sample
    Returns:
        df_sample (Dask DataFrame): retrieved sample.
    """
