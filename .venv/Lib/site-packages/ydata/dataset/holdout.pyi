from _typeshed import Incomplete
from dask.dataframe import DataFrame as DataFrame
from typing import Literal
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.random import RandomSeed as RandomSeed

def check_unique_index(dataset: DataFrame): ...

class Holdout:
    """Class responsible for selecting an holdout/test subset from a given
    Dataset. Stores the holdout indexes.

    Arguments:

    Properties:
        holdout_index (): Holdout index
    """
    uuid: Incomplete
    def __init__(self, fraction: float = 0.2) -> None:
        """Initialize Holdout class."""
    def get_split(self, X: Dataset, metadata: Metadata | None = None, random_state: RandomSeed = None, strategy: Literal['random', 'stratified'] = 'random') -> tuple:
        """Returns the indexes of the holdout portion."""
    @property
    def holdout_def(self):
        """Returns the holdout divisitions and npartitions to be loaded while
        loading the same Dataset again."""
    def save(self, path: str):
        """Saves the Holdout object."""
    @classmethod
    def load(cls, path: str):
        """Load the Holdout object."""
    def __len__(self) -> int: ...
