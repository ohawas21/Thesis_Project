"""Data Connectors utils to sample the read Dask dataframe."""
from math import ceil
from typing import Union

from dask import dataframe as dddataframe


def sample_fraction(df, sample: Union[float, int], total_rows: int = None):
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
    if sample >= 1:
        # size is either provided (total_rows) or calculated (nrows(df))
        size = total_rows if total_rows else len(df)
        return sample / size
    elif 0 < sample < 1:  # if pct of total
        return sample
    else:
        raise Exception(f"Requested sample ({sample}) is not valid.")


def nsample(df: dddataframe, sample_size: int) -> dddataframe:
    """Obtain a contiguous sample from a Dask DataFrame with sample_size
    records.

    Args:
        df: original dataframe
        sample_size (int): requested number of rows for the sample
    Returns:
        df_sample (Dask DataFrame): retrieved sample.
    """
    # Get the number of partitions
    n_partitions = df.npartitions
    part_size = len(df.partitions[0])
    req_parts = sample_size / part_size

    if n_partitions * part_size <= sample_size:
        return df
    elif req_parts < 1:
        sample = df.head(n=sample_size)
    elif req_parts == 1:
        sample = df.partitions[0]
    else:
        sample = df.head(sample_size, npartitions=ceil(req_parts))
    return sample
