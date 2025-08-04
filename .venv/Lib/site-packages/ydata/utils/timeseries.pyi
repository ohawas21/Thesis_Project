from pandas import DataFrame as DataFrame

def has_datetimeindex(func):
    """Guarantees a function is called on a DataFrame with a DatetimeIndex."""
def add_temporal_features(df: DataFrame, feats: str | list[str]):
    """Calculates features based on timestamp.

    Pandas features: all attributes available from df.index.{feat}
    Custom features: 'month-hour', 'time-of-day', 'quarter-hour', 'quarter-tod'

    Args:
        df (DataFrame)
        feats (Union[str, List[str]])

    Returns
        df (DataFrame): original data w/ new temporal features as columns
    """
