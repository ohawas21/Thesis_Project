from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame, Series as pdSeries

def spearman(df: pdDataFrame) -> float:
    """calculate Spearman's correlation for a dataframe.

    Args:
        df (pdDataFrame): dataframe

    Returns:
        float: Spearman's score
    """
def cramers_v(col_1: pdSeries, col_2: pdSeries) -> float:
    """calculate Cramer's V between two pandas series.

    Args:
        col_1 (pdSeries): first column to consider
        col_2 (pdSeries): second column to consider

    Returns:
        float: Cramer's V score
    """

mapping: Incomplete
