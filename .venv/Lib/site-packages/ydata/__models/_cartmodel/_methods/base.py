"""Defined the base class for the CartHierarchical valid methods."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

from numpy import clip, max, min

from ydata.dataset.dataset import Dataset
from ydata.utils.data_types import DataType, VariableType
from ydata.utils.random import RandomSeed

# TODO revise here the use of the data types


class BaseMethod(ABC):
    """Base class for the Methods supported by CartHierarchical."""

    def __init__(self, random_state: RandomSeed = None):
        """Initialize BaseMethod.

        Args:
            random_state (int): Internal random state.
        """
        self.random_state = random_state

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def prepare_dfs(
        self,
        X: Dataset,
        y: Optional[Dataset] = None,
        dtypes: Dict = None,
        normalise_num_cols: bool = True,
        fit: bool = True,
    ):
        X = X.copy()
        if y is not None and self.y_dtype == DataType.NUMERICAL:
            y = y.copy()
            not_nan_indices = y.notna()
            X = X.loc[not_nan_indices]
            y = y.loc[not_nan_indices]

        if normalise_num_cols:
            if fit:
                num_cols = [
                    k
                    for k, v in dtypes.items()
                    if v["datatype"] == DataType.NUMERICAL
                    and k in X.columns
                    and v["vartype"] != VariableType.DATETIME
                ]
                self.num_cols_range = {}
                for col in num_cols:
                    self.num_cols_range[col] = {
                        "min": min(X[col]), "max": max(X[col])}
                    if (
                        self.num_cols_range[col]["max"]
                        == self.num_cols_range[col]["min"]
                    ):
                        self.num_cols_range[col]["ratio"] = 1.0
                    else:
                        self.num_cols_range[col]["ratio"] = (
                            self.num_cols_range[col]["max"]
                            - self.num_cols_range[col]["min"]
                        )
                    X[col] = (
                        X[col] - self.num_cols_range[col]["min"]
                    ) / self.num_cols_range[col]["ratio"]

            else:
                for col in self.num_cols_range:
                    X[col] = (
                        X[col] - self.num_cols_range[col]["min"]
                    ) / self.num_cols_range[col]["ratio"]
                    X[col] = clip(X[col], 0, 1)
        return X, y
