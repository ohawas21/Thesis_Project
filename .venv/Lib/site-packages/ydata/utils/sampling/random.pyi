from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.dask import DaskCluster
from ydata.utils.random import RandomSeed as RandomSeed

class RandomPartitionSampler:
    def __init__(self, cluster: DaskCluster = None) -> None: ...
    def sample(self, dataset: Dataset, frac: float, return_non_sampled: bool = False, metadata: Metadata | None = None, columns: dict | None = None, random_state: RandomSeed = None) -> Dataset | tuple[Dataset, Dataset]: ...

class RandomSplitSampler:
    def sample(self, dataset: Dataset, frac: float, return_non_sampled: bool = False, metadata: Metadata | None = None, columns: dict | None = None, random_state: RandomSeed = None) -> Dataset | tuple[Dataset, Dataset]: ...
