"""
    Stratified sampler definition
"""
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

from typing import Optional

import dask.dataframe as ddDataframe
from dask.dataframe.utils import make_meta
from numpy import ceil as np_ceil
from numpy.random import default_rng
from pandas import DataFrame as pdDataFrame

from ydata.dataset.dataset import Dataset
from ydata.metadata.metadata import Metadata
from ydata.utils.data_types import DataType, VariableType
from ydata.utils.discretize import DiscretizationType, Discretizer
from ydata.utils.random import RandomSeed
from ydata.utils.validations import validate_columns_in_dataset


class StratifiedSampler:
    def __init__(
        self,
        columns: Optional[dict] = None,
        n_bins: int = 10,
        diversity_threshold: float = 0.8,
        discretization_type: DiscretizationType = DiscretizationType.QUANTILE,
    ):
        self.datatypes = None
        self.columns = columns
        self.column_skip = set()
        self.columns_to_be_sampled = set()
        self.diversity_threshold = diversity_threshold
        self._discretizer = Discretizer(
            n_bins=n_bins, method=discretization_type, reset_index=False
        )
        self.all_columns = []
        self.sample_index = None
        self.random_state = None

    def sample(
        self,
        dataset: Dataset,
        frac: float,
        return_non_sampled: bool = False,
        metadata: Metadata | None = None,
        columns: dict | None = None,
        random_state: RandomSeed = None,
    ) -> Dataset | tuple[Dataset, Dataset]:
        self.random_state = random_state
        validate_columns_in_dataset(dataset=dataset, columns=columns)

        if frac >= 1:
            holdout_ds = Dataset(dataset._data.sample(
                frac=1., random_state=self.random_state))
        else:
            self.columns = columns
            if self.columns is None:
                if metadata is not None:
                    self.columns = metadata.columns.keys()
                else:
                    self.columns = dataset.columns
            self.all_columns = dataset.columns
            self._set_metadata(dataset, metadata)
            meta_mask = make_meta(dataset._data)
            samples = dataset._data.map_partitions(
                self._sample_partition, frac, meta=meta_mask
            )
            holdout_ds = Dataset(samples)

        if return_non_sampled:
            holdout_mask = dataset._data.index.isin(
                list(holdout_ds._data.index))
            train_ds = Dataset(dataset._data.loc[~holdout_mask])
            return holdout_ds, train_ds
        return holdout_ds

    def _sample_partition(self, dataframe: pdDataFrame, frac: float) -> ddDataframe:

        samples_per_column = int(
            np_ceil(frac * len(dataframe) / len(self.columns_to_be_sampled))
        )
        valid_datatypes = {
            key: self.datatypes[key]
            for key in self.datatypes.keys() & self.columns_to_be_sampled
        }
        dataset = self._discretizer.discretize_dataframe(
            dataframe, valid_datatypes
        )
        sampler = set()
        for column in self.columns_to_be_sampled:
            aggregated_indexes = self._aggregate_dataset_to_lists(
                dataset, column, random_state=self.random_state
            )
            list_lengths = self._get_list_lengths(aggregated_indexes)
            sampling_strategy = self._get_proportional_lists(
                list_lengths, samples_per_column
            )
            samples = self._add_rows_to_sample(
                sampling_strategy, aggregated_indexes)
            sampler.update(samples)
        dataset = self.extract_sample(dataframe, sampler)
        self.sample_index = dataset.index
        return dataset

    @staticmethod
    def _get_proportional_lists(list_lengths: list, samples_per_column: int) -> dict:
        choices = {}
        acc = 0
        threshold = 0.8
        for i, n_records in enumerate(list_lengths):
            n_records_to_sample = (
                samples_per_column * n_records / sum(list_lengths)
            )
            acc += n_records_to_sample
            if acc >= threshold:
                choices[i] = int(np_ceil(n_records_to_sample))
                acc = 0
            else:
                choices[i] = 0

        return choices

    def extract_sample(self, dataframe: pdDataFrame, samples: set) -> pdDataFrame:
        indexes = list(samples)
        dataframe = dataframe.loc[indexes]
        dataframe = dataframe[self.all_columns]
        return (
            dataframe if not dataframe.empty else pdDataFrame(
                columns=self.all_columns)
        )

    @staticmethod
    def _aggregate_dataset_to_lists(df: pdDataFrame, columns, random_state: RandomSeed = None) -> list:
        rng = default_rng(seed=random_state)

        def _shuffle(x):
            rng.shuffle(x)
            return x

        aggregated_df = df.reset_index().groupby(columns).agg({"index": list})
        return aggregated_df["index"].apply(_shuffle).tolist()

    @staticmethod
    def _get_list_lengths(whole_lists: list) -> list:
        return [len(index_lists) for index_lists in whole_lists]

    @staticmethod
    def _add_rows_to_sample(sampling_strategy: dict, row_index_list: list) -> set:
        sampler = set()
        for key, no_samples in sampling_strategy.items():
            sampler.update(row_index_list[key][:no_samples])
        return sampler

    def _set_columns_to_sample(self) -> None:
        # on the basis of the diversity threshold, the number of unique values/ the number of values
        num_cols = [
            col for col in self.columns if self.datatypes[col] == DataType.NUMERICAL
        ]
        non_num_col = [col for col in self.columns if col not in num_cols]

        self.columns_to_be_sampled.update(num_cols)

        for column in non_num_col:
            diversity = self.n_unique[column] / self.nrows
            if diversity <= self.diversity_threshold:
                self.columns_to_be_sampled.add(column)

        if len(self.columns_to_be_sampled) == 0:
            self.columns_to_be_sampled.add(self.columns[0])

    def _set_metadata(
        self, dataset: Dataset, metadata: Optional[Metadata] = None
    ) -> None:
        if metadata is None:
            self.nrows = dataset.nrows
            self.datatypes, self.n_unique = self._estimate_datatypes_and_unique(
                dataset)
            self._set_columns_to_sample()
        else:
            self.datatypes = {k: v.datatype for k,
                              v in metadata.columns.items()}
            self.n_unique = metadata.summary["cardinality"]
            self.nrows = metadata.summary["nrows"]
            self._set_columns_to_sample()

    def _estimate_datatypes_and_unique(self, dataset: Dataset) -> dict:
        datatype = {}
        n_unique = {}

        for column_name, variable_type in dataset.schema.items():
            n_unique_values = dataset._data[column_name].nunique(
                dropna=False).compute()
            n_unique[column_name] = n_unique_values
            # this appears to be the definition of numerical type after reading the metadata file
            if variable_type in [VariableType.INT, VariableType.FLOAT]:
                datatype[column_name] = DataType.NUMERICAL
            else:
                datatype[column_name] = "OTHER"

        return datatype, n_unique
