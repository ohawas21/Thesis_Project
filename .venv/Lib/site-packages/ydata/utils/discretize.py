import warnings
from enum import Enum
from typing import Dict, Optional

from numpy import number as np_number
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
from pandas import cut as pdcut
from pandas import qcut as pdqcut

from ydata.utils.data_types import DataType


class DiscretizationType(Enum):
    UNIFORM = "uniform"
    QUANTILE = "quantile"


class Discretizer:

    """A class which enables the discretization of a pandas dataframe. Perform
    this action when you want to convert a continuous variable into a
    categorical variable.

    Attributes:

    method (DiscretizationType): this attribute controls how the buckets of your
    discretization are formed. A uniform discretization type forms the bins to be of
    equal width whereas a quantile discretization type forms the bins to be of equal size.

    n_bins (int): number of bins
    reset_index (bool): instruction to reset the index of the dataframe after the discretization
    """

    def __init__(
        self, method=DiscretizationType, n_bins: int = 10, reset_index: bool = False
    ) -> None:

        self.discretization_type = method
        self.n_bins = n_bins
        self.reset_index = reset_index

    def discretize_dataframe(
        self, dataframe: pdDataFrame, data_types: Optional[Dict[str, DataType]] = None
    ) -> pdDataFrame:
        """discretize the dataframe.

        Args:
            dataframe (pdDataFrame): dataframe to be discretized
            data_types (Optional[dict], optional): dictionary specifying datatypes. Defaults to None.

        Returns:
            pdDataFrame: discretized dataframe
        """

        discretized_df = dataframe.copy()
        all_columns = dataframe.columns
        num_columns = self._get_numerical_columns(dataframe, data_types)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            for column in num_columns:
                discretized_df.loc[:, column] = self._discretize_column(
                    discretized_df[column]
                )

        discretized_df = discretized_df[all_columns]
        return (
            discretized_df.reset_index(drop=True)
            if self.reset_index
            else discretized_df
        )

    def _discretize_column(self, column: pdSeries) -> pdSeries:
        if self.discretization_type == DiscretizationType.UNIFORM:
            return self._descritize_quantile(column)

        elif self.discretization_type == DiscretizationType.QUANTILE:
            return self._descritize_uniform(column)

    def _descritize_quantile(self, column: pdSeries) -> pdSeries:
        return pdqcut(
            column, q=self.n_bins, labels=False, retbins=False, duplicates="drop"
        ).values

    def _descritize_uniform(self, column: pdSeries) -> pdSeries:
        return pdcut(
            column, bins=self.n_bins, labels=False, retbins=True, duplicates="drop"
        )[0].values

    def _get_numerical_columns(self, dataframe: pdDataFrame, data_types: dict) -> list:
        if data_types is not None:
            return [col for col in data_types if data_types[col] == DataType.NUMERICAL and dataframe[col].value_counts().sum() > 1]
        else:
            return dataframe.select_dtypes(include=np_number).columns.tolist()
