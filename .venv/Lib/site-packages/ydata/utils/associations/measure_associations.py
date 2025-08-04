from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from dask import delayed
from numpy import ones as np_ones
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries

from ydata.dataset import Dataset
from ydata.utils.data_types import DataType, ScaleType, VariableType
from ydata.utils.discretize import DiscretizationType, Discretizer
from ydata.utils.validations import validate_columns_in_dataset


class MeasureAssociations:
    """A class that is able to compute pairwise column scores using the methods
    specified by the mapping provided in the initialisation."""

    def __init__(
        self,
        mapping: PairwiseMatrixMapping,
        n_bins: int = 10,
    ) -> None:
        self.mapping = mapping
        self.discretizer = Discretizer(
            method=DiscretizationType.UNIFORM, n_bins=n_bins, reset_index=False
        )

    def compute_pairwise_matrix(
        self,
        dataset: Dataset | pdDataFrame,
        datatypes: dict = None,
        vartypes: dict | None = None,
        columns: list[str] | None = None
    ) -> pdDataFrame:
        """Compute the pairwise column matrix given the specified mapping.

        Args:
            dataset (Dataset): dataset
            columns (list[str] | None): columns. Defaults to None.
        Returns:
            pdDataFrame: pairwise column matrix
        """
        validate_columns_in_dataset(dataset, columns)
        self.columns = columns if columns is not None else dataset.columns
        self.vartypes = dataset.schema if vartypes is None else vartypes
        if datatypes is None:
            self.datatypes = self._estimate_datatypes(self.vartypes)
        else:
            self.datatypes = datatypes

        scale_types = self._get_scale_types(self.datatypes, self.vartypes)
        self.column_pairs = self._get_column_pairs(scale_types)

        data = dataset._data if isinstance(dataset, Dataset) else dataset
        partition_results = data.map_partitions(
            self._calculate_partition_relationships, meta={}
        )
        return self._relationships_to_df(partition_results)

    def _get_scale_types(
        self, data_types: dict[str, DataType], variable_types: dict[str, VariableType]
    ) -> list[ColumnMetric]:
        """Calculate the scale types based on the column's variable and data
        type.

        Args:
            data_types (dict): column data types
            variable_types (dict): column variable types

        Returns:
            list[ColumnMetric]: a list of a ColumnMetric instance with a scale type attribute
        """
        return [ColumnMetric(column_name, data_types[column_name], variable_types[column_name]) for column_name in data_types.keys()]

    @staticmethod
    def _get_column_pairs(
        column_data: list[ColumnMetric],
    ) -> list[tuple[ColumnMetric, ColumnMetric]]:
        """Find combination of every column pair.

        Args:
            column_data (list[str]): list of ColumnMetric instances

        Returns:
            list[tuple[ColumnMetric, ColumnMetric]]: a list of all ColumnMetric combinations
        """
        column_pairs = []
        for subset in itertools.combinations(column_data, 2):
            if subset[0].variable_type not in [VariableType.DATETIME, VariableType.DATE] and subset[1].variable_type not in [VariableType.DATETIME, VariableType.DATE]:
                column_pairs.append(subset)
        return column_pairs

    def _calculate_partition_relationships(self, dataframe: pdDataFrame) -> dict[tuple, float]:
        """Calculate the score for each relationship at each dataframe
        partition of the dataset.

        Args:
            dataframe (pdDataFrame): dataframe
        Returns:
            dict: dictionary specifying the score for each ColumnMetric pair.
        """
        relationships = {}
        if len(dataframe) > 3:
            discretized_dataframe = self.discretizer.discretize_dataframe(
                dataframe.copy(), data_types=self.datatypes
            )
            combined_calculation_columns = defaultdict(set)
            for col_1, col_2 in self.column_pairs:
                task = self.mapping.mapping[(
                    col_1.scale_type, col_2.scale_type)]
                if task.params.combined_calculation:
                    combined_calculation_columns[task.method].update(
                        {col_1.name, col_2.name})
                else:
                    params = self.resolve_params(
                        col_1, col_2, task.params, dataframe, discretized_dataframe)
                    relationships[(col_1.name, col_2.name)
                                  ] = task.method(**params)
            # it is much faster to calculate some columnwise metrics together.
            for combined_task, columns in combined_calculation_columns.items():
                combined_score = combined_task(dataframe[list(columns)])
                for subset in itertools.combinations(list(columns), 2):
                    relationships[subset] = combined_score[subset[0]][subset[1]]

        return relationships

    @staticmethod
    def resolve_params(
        col_1: ColumnMetric,
        col_2: ColumnMetric,
        params: PairwiseMatrixMapping,
        df: pdDataFrame,
        df_discretized: pdDataFrame,
    ) -> dict:
        """select the relevant dataframe for the function based on the
        mapping."""

        def f(x: str) -> pdSeries:
            return df_discretized if x else df
        return {
            "col_1": f(params.discretize[0])[col_1.name],
            "col_2": f(params.discretize[1])[col_2.name],
        }

    def _estimate_datatypes(self, vartypes: dict) -> dict:
        """estimate the datatypes of the dataset."""
        datatype = {}
        for column_name, variable_type in vartypes.items():
            # this appears to be the definition of numerical type after reading the metadata file
            if variable_type in [VariableType.INT, VariableType.FLOAT, VariableType.DATE, VariableType.DATETIME]:
                datatype[column_name] = DataType.NUMERICAL
            else:
                datatype[column_name] = DataType.CATEGORICAL
        return datatype

    @delayed
    def _relationships_to_df(self, relationships):
        relationships = pdDataFrame(dict(relationships)).mean(axis=1).to_dict()
        return self._relationships_to_matrix(relationships)

    def _relationships_to_matrix(
        self, relationships,
    ) -> pdDataFrame:
        """convert the relationships dictionary into a matrix using a pandas
        dataframe."""
        array = np_ones((len(self.columns), len(self.columns)))
        df = pdDataFrame(array, index=self.columns, columns=self.columns)
        for (col_1_name, col_2_name), score in relationships.items():
            df.at[col_1_name, col_2_name] = score
            df.at[col_2_name, col_1_name] = score
        return df


class ColumnMetric:
    def __init__(
        self, column_name: str, data_type: DataType, variable_type: VariableType
    ) -> None:
        """class responsible for holding key information about the column for
        metric caluculations."""
        self.name = column_name
        self.data_type = data_type
        self.variable_type = variable_type
        self.scale_type = self._get_scale_type(data_type, variable_type)

    @staticmethod
    def _get_scale_type(data_type: DataType, variable_type: VariableType) -> None:
        """infer the scale type based on the data and variable type of a
        column."""
        if data_type == DataType.NUMERICAL:
            scale_type = ScaleType.ORDINAL

        elif data_type == DataType.CATEGORICAL:
            if variable_type in [VariableType.INT, VariableType.FLOAT]:
                scale_type = ScaleType.ORDINAL
            else:
                scale_type = ScaleType.NOMINAL
        elif data_type == DataType.STR:
            scale_type = ScaleType.NOMINAL
        else:
            scale_type = ScaleType.ORDINAL
        return scale_type


class MeasureAssociationsPandas(MeasureAssociations):
    def compute_pairwise_matrix(
        self,
        dataframe: pdDataFrame,
        datatypes: dict,
        vartypes: dict,
        columns: list[str] | None = None,
    ) -> pdDataFrame:
        """compute the pairwise column matrix for a pandas dataframe given the
        specified mapping."""

        self.datatypes = datatypes
        self.vartypes = vartypes

        self.columns = columns if columns is not None else dataframe.columns

        scale_types = self._get_scale_types(datatypes, vartypes)
        self.column_pairs = self._get_column_pairs(scale_types)
        associations_dict = self._calculate_partition_relationships(dataframe)
        return self._relationships_to_matrix(associations_dict)


@dataclass
class PairwiseTask:
    """dataclass holding details about one part of the pairwise mapping."""

    method: Callable[[pdSeries, pdSeries], float]
    params: PairwiseParams


@dataclass
class PairwiseParams:
    """dataclass specifying the particular parameters of a pairwise task."""

    discretize: tuple[bool, bool]
    combined_calculation: bool


@dataclass
class PairwiseMatrixMapping:
    """dataclass holding the rules of the mapping, linking scale types to
    pairwise tasks."""

    mapping: dict[tuple[ScaleType, ScaleType], PairwiseTask]
