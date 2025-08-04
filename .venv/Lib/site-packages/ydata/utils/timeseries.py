"""Utilities for dealing with time-series."""
from functools import wraps
from typing import List, Union

from pandas import DataFrame, DatetimeIndex, cut


def has_datetimeindex(func):
    "Guarantees a function is called on a DataFrame with a DatetimeIndex."

    @wraps(func)
    def wrapper(df, *args, **kwargs):
        assert isinstance(
            df.index, DatetimeIndex
        ), "DataFrame index must be a pd.DatetimeIndex"
        return func(df, *args, **kwargs)

    return wrapper


@has_datetimeindex
def add_temporal_features(df: DataFrame, feats: Union[str, List[str]]):
    """Calculates features based on timestamp.

    Pandas features: all attributes available from df.index.{feat}
    Custom features: 'month-hour', 'time-of-day', 'quarter-hour', 'quarter-tod'

    Args:
        df (DataFrame)
        feats (Union[str, List[str]])

    Returns
        df (DataFrame): original data w/ new temporal features as columns
    """
    df = df.copy()
    feats = [feats] if isinstance(feats, str) else feats  # convert to list
    for feat in feats:
        if (
            feat == "time-of-day"
        ):  # Time-of-Day: Morning (0H:8H), Noon (8H, 16H), Evening (16H, 24H)
            df[feat] = cut(
                df.index.hour,
                [0, 7, 15, 24],
                labels=["morning", "noon", "evening"],
                include_lowest=True,
            )
        elif feat == "quarter-hour":  # Quarter + Hour of Day (example: Q1_H08)
            df[feat] = (
                "Q"
                + df.index.quarter.astype(str)
                + "_H"
                + df.index.hour.astype(str).str.zfill(2)
            )
        elif feat == "quarter-tod":  # Quarter + Time-of-Day
            df[feat] = (
                "Q"
                + df.index.quarter.astype(str)
                + "_"
                + cut(
                    df.index.hour,
                    [0, 7, 15, 24],
                    labels=["morning", "noon", "evening"],
                    include_lowest=True,
                ).astype(str)
            )
        elif feat == "month-hour":
            df[feat] = (
                "M"
                + df.index.month.astype(str)
                + "_H"
                + df.index.hour.astype(str).str.zfill(2)
            )
        else:
            try:  # Enable default aggregations of pandas (e.g. hour, quarter)
                df[feat] = getattr(df.index, feat)
            except BaseException:
                raise NotImplementedError(
                    f"The specified temporal aggregation {feat} is not supported."
                )
    return df
