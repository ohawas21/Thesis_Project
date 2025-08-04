from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.dataset.dataset import Dataset
from ydata.metadata.metadata import Metadata
from ydata.utils.discretize import DiscretizationType
from ydata.utils.random import RandomSeed as RandomSeed

class StratifiedSampler:
    datatypes: Incomplete
    columns: Incomplete
    column_skip: Incomplete
    columns_to_be_sampled: Incomplete
    diversity_threshold: Incomplete
    all_columns: Incomplete
    sample_index: Incomplete
    random_state: Incomplete
    def __init__(self, columns: dict | None = None, n_bins: int = 10, diversity_threshold: float = 0.8, discretization_type: DiscretizationType = ...) -> None: ...
    def sample(self, dataset: Dataset, frac: float, return_non_sampled: bool = False, metadata: Metadata | None = None, columns: dict | None = None, random_state: RandomSeed = None) -> Dataset | tuple[Dataset, Dataset]: ...
    def extract_sample(self, dataframe: pdDataFrame, samples: set) -> pdDataFrame: ...
