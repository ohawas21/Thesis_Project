import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from ydata.dataset.dataset import Dataset as Dataset
from ydata.utils.random import RandomSeed as RandomSeed

class BaseMethod(ABC, metaclass=abc.ABCMeta):
    """Base class for the Methods supported by CartHierarchical."""
    random_state: Incomplete
    def __init__(self, random_state: RandomSeed = None) -> None:
        """Initialize BaseMethod.

        Args:
            random_state (int): Internal random state.
        """
    @abstractmethod
    def fit(self): ...
    @abstractmethod
    def predict(self): ...
    num_cols_range: Incomplete
    def prepare_dfs(self, X: Dataset, y: Dataset | None = None, dtypes: dict = None, normalise_num_cols: bool = True, fit: bool = True): ...
