from _typeshed import Incomplete
from dataclasses import dataclass
from pandas import DataFrame as pdDataFrame, Series as pdSeries
from typing import Callable
from ydata.dataset import Dataset
from ydata.utils.data_types import DataType, ScaleType, VariableType

class MeasureAssociations:
    """A class that is able to compute pairwise column scores using the methods
    specified by the mapping provided in the initialisation."""
    mapping: Incomplete
    discretizer: Incomplete
    def __init__(self, mapping: PairwiseMatrixMapping, n_bins: int = 10) -> None: ...
    columns: Incomplete
    vartypes: Incomplete
    datatypes: Incomplete
    column_pairs: Incomplete
    def compute_pairwise_matrix(self, dataset: Dataset | pdDataFrame, datatypes: dict = None, vartypes: dict | None = None, columns: list[str] | None = None) -> pdDataFrame:
        """Compute the pairwise column matrix given the specified mapping.

        Args:
            dataset (Dataset): dataset
            columns (list[str] | None): columns. Defaults to None.
        Returns:
            pdDataFrame: pairwise column matrix
        """
    @staticmethod
    def resolve_params(col_1: ColumnMetric, col_2: ColumnMetric, params: PairwiseMatrixMapping, df: pdDataFrame, df_discretized: pdDataFrame) -> dict:
        """select the relevant dataframe for the function based on the
        mapping."""

class ColumnMetric:
    name: Incomplete
    data_type: Incomplete
    variable_type: Incomplete
    scale_type: Incomplete
    def __init__(self, column_name: str, data_type: DataType, variable_type: VariableType) -> None:
        """class responsible for holding key information about the column for
        metric caluculations."""

class MeasureAssociationsPandas(MeasureAssociations):
    datatypes: Incomplete
    vartypes: Incomplete
    columns: Incomplete
    column_pairs: Incomplete
    def compute_pairwise_matrix(self, dataframe: pdDataFrame, datatypes: dict, vartypes: dict, columns: list[str] | None = None) -> pdDataFrame:
        """compute the pairwise column matrix for a pandas dataframe given the
        specified mapping."""

@dataclass
class PairwiseTask:
    """dataclass holding details about one part of the pairwise mapping."""
    method: Callable[[pdSeries, pdSeries], float]
    params: PairwiseParams
    def __init__(self, method, params) -> None: ...

@dataclass
class PairwiseParams:
    """dataclass specifying the particular parameters of a pairwise task."""
    discretize: tuple[bool, bool]
    combined_calculation: bool
    def __init__(self, discretize, combined_calculation) -> None: ...

@dataclass
class PairwiseMatrixMapping:
    """dataclass holding the rules of the mapping, linking scale types to
    pairwise tasks."""
    mapping: dict[tuple[ScaleType, ScaleType], PairwiseTask]
    def __init__(self, mapping) -> None: ...
