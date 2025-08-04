from itertools import product

from dask import compute
from dask.dataframe.core import DataFrame as ddDataFrame
from numpy import arange as np_arange
from pandas import DataFrame as pdDataFrame
from pandas import MultiIndex as pdMultiIndex

from ydata.core.enum import StringEnum
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.dask import DaskCluster


class InteractionType(StringEnum):
    RECTANGULAR_BINNING = "rectangular"


class InteractionEngine:
    def __init__(self) -> None:
        self._cluster = DaskCluster()

    def _preprocess(self, dataset: Dataset, metadata: Metadata, num_intervals: int) -> ddDataFrame:
        dask_df = dataset.to_dask().copy()
        domains = metadata.summary["domains"]
        for col in metadata.numerical_vars:
            dask_df[col] = (dask_df[col] - domains[col]["min"]) / (
                domains[col]["max"] - domains[col]["min"])
            dask_df[col] = (
                dask_df[col] * (num_intervals - 1.0)).astype(int)
        return dask_df

    @staticmethod
    def _calculate_rectangular_binning(params: tuple[ddDataFrame, int, int]) -> dict:
        df, col1_ix, num_intervals = params
        col_1 = df.columns[col1_ix]
        df_groupby_tasks = {col_2: df.groupby(
            [col_1, col_2]).size() for col_2 in df.columns[col1_ix+1:]}
        df_groupby_res = compute(df_groupby_tasks)[0]

        x_y_pos = np_arange(num_intervals)
        index_values = list(product(x_y_pos, x_y_pos))

        interactions_col1 = {}
        for col_2 in df.columns[col1_ix+1:]:
            index = pdMultiIndex.from_tuples(
                index_values, names=[col_1, col_2])
            df_interactions = pdDataFrame(0, index=index, columns=["z"])
            df_interactions.loc[df_groupby_res[col_2].index,
                                "z"] = df_groupby_res[col_2]
            df_interactions = df_interactions.reset_index().rename(
                columns={col_1: "x", col_2: "y"})
            interactions_col1[col_2] = list(
                df_interactions.to_dict("index").values())
        return interactions_col1

    def calculate(self, dataset: Dataset, metadata: Metadata,
                  interaction_type: InteractionType | str = InteractionType.RECTANGULAR_BINNING,
                  num_intervals: int = 15) -> list:
        if not metadata.numerical_vars:
            return []

        interaction_fn = None
        if interaction_type == InteractionType.RECTANGULAR_BINNING:
            interaction_fn = self._calculate_rectangular_binning
        else:
            raise ValueError(
                f"Invalid interaction type. The valid types are: {', '.join([t.value for t in InteractionType])}.")

        dask_df = self._preprocess(dataset, metadata, num_intervals)

        return self._cluster.client.map(
            interaction_fn,
            [(dask_df[metadata.numerical_vars], i, num_intervals)
             for i in range(len(metadata.numerical_vars) - 1)],
            key=metadata.numerical_vars[:-1])
