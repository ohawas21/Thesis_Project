
import numpy as np
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries

from ydata.utils.associations.measure_associations import PairwiseMatrixMapping, PairwiseParams, PairwiseTask
from ydata.utils.associations.metrics.cramers_v import compute_adjusted_cramers_v, compute_chi_squared_based_values
from ydata.utils.data_types import ScaleType


def spearman(df: pdDataFrame) -> float:
    """calculate Spearman's correlation for a dataframe.

    Args:
        df (pdDataFrame): dataframe

    Returns:
        float: Spearman's score
    """
    return df.corr(method="spearman")


def cramers_v(col_1: pdSeries, col_2: pdSeries) -> float:
    """calculate Cramer's V between two pandas series.

    Args:
        col_1 (pdSeries): first column to consider
        col_2 (pdSeries): second column to consider

    Returns:
        float: Cramer's V score
    """
    cross_t_shape, chi2, sample_size = compute_chi_squared_based_values(
        col_1, col_2)
    if cross_t_shape is None:
        return np.nan
    return compute_adjusted_cramers_v(cross_t_shape, chi2, sample_size)


mapping = PairwiseMatrixMapping(
    {
        (ScaleType.ORDINAL, ScaleType.ORDINAL): PairwiseTask(
            spearman, params=PairwiseParams(discretize=(
                False, False), combined_calculation=True)
        ),
        (ScaleType.ORDINAL, ScaleType.NOMINAL): PairwiseTask(
            cramers_v, params=PairwiseParams(discretize=(
                True, False), combined_calculation=False)
        ),
        (ScaleType.NOMINAL, ScaleType.ORDINAL): PairwiseTask(
            cramers_v, params=PairwiseParams(discretize=(
                False, True), combined_calculation=False)
        ),
        (ScaleType.NOMINAL, ScaleType.NOMINAL): PairwiseTask(
            cramers_v, params=PairwiseParams(discretize=(
                False, False), combined_calculation=False)
        ),
    }
)
