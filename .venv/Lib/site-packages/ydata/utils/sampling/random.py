from dask.dataframe import concat as ddConcat
from dask.dataframe.core import DataFrame as ddDataFrame
from dask.dataframe.io import from_pandas as dd_from_pandas
from dask.distributed import as_completed
from numpy import repeat as np_repeat

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.dask import DaskCluster
from ydata.utils.random import RandomSeed


class RandomPartitionSampler:

    def __init__(self, cluster: DaskCluster = None) -> None:
        self._cluster = cluster

    def _sample_for_one_partition(self,
                                  dataset: Dataset,
                                  frac: float,
                                  return_non_sampled: bool = False,
                                  random_state: RandomSeed = None
                                  ) -> Dataset | tuple[Dataset, Dataset]:
        holdout_dask = dataset._data.sample(frac=frac, random_state=random_state)
        holdout_mask = dataset._data.index.isin(list(holdout_dask.index))
        train_dask = dataset._data.loc[~holdout_mask]
        holdout_ds = Dataset(holdout_dask)
        if return_non_sampled:
            train_ds = Dataset(train_dask)
            return holdout_ds, train_ds
        return holdout_ds

    def sample(self,
               dataset: Dataset,
               frac: float,
               return_non_sampled: bool = False,
               metadata: Metadata | None = None,
               columns: dict | None = None,
               random_state: RandomSeed = None,
               ) -> Dataset | tuple[Dataset, Dataset]:

        return self._sample_for_one_partition(dataset, frac, return_non_sampled, random_state=random_state)

        def map_sample(dask_df: ddDataFrame, fraction: float) -> [ddDataFrame, ddDataFrame]:
            df = dask_df.compute()
            holdout_df = df.sample(frac=fraction)
            train_df = df[~df.index.isin(holdout_df.index)]
            return dd_from_pandas(holdout_df, npartitions=1), \
                dd_from_pandas(train_df, npartitions=1)

        if self._cluster is None:
            self._cluster = DaskCluster()

        # Ensure the cluster has the minimum of workers active, otherwise the scatter will fail.
        self._cluster.adapt(wait=True)

        keys = [f"{p_id}" for p_id in range(dataset._data.npartitions)]
        map_params = {
            "dask_df": [self._cluster.client.scatter(dataset._data.get_partition(p_id))
                        for p_id in range(dataset._data.npartitions)],
            "fraction": np_repeat(frac, dataset._data.npartitions).tolist()
        }

        dask_futures = self._cluster.client.map(map_sample,
                                                *map_params.values(),
                                                key=keys)

        holdout_partitions = []
        train_partitions = []
        for _, partition_holdout in as_completed(dask_futures, with_results=True):
            holdout_dask, train_dask = partition_holdout
            holdout_partitions.append(holdout_dask)
            if return_non_sampled:
                train_partitions.append(train_dask)

        holdout_ds = Dataset(ddConcat(holdout_partitions, axis=0))
        if return_non_sampled:
            train_ds = Dataset(ddConcat(train_partitions, axis=0))
            return holdout_ds, train_ds

        return holdout_ds


class RandomSplitSampler:

    def sample(self,
               dataset: Dataset,
               frac: float,
               return_non_sampled: bool = False,
               metadata: Metadata | None = None,
               columns: dict | None = None,
               random_state: RandomSeed = None
               ) -> Dataset | tuple[Dataset, Dataset]:

        train_dask, holdout_dask = dataset._data.random_split(
            [1.0 - frac, frac], shuffle=False, random_state=random_state)

        holdout_ds = Dataset(holdout_dask)
        if return_non_sampled:
            train_ds = Dataset(train_dask)
            return holdout_ds, train_ds

        return holdout_ds
