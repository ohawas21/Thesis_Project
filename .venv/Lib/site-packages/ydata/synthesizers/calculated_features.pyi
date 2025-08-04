from numpy import ndarray as ndarray
from pandas import DataFrame as pdDataFrame, Series as pdSeries
from typing import Callable
from ydata.dataset.dataset import Dataset as Dataset

class CalculatedFeature:
    def __init__(self, features: str | list[str], function: Callable, calculated_from: list[str] | str, reference_keys: dict[str, list | str] | None = None) -> None: ...
    @property
    def features(self) -> list[str]: ...
    @property
    def function(self) -> Callable: ...
    @property
    def calculated_from(self) -> list[str]: ...
    @property
    def reference_keys(self) -> list[tuple[str]]: ...
    def apply_to(self, dataframe: pdDataFrame) -> pdSeries | ndarray | tuple[pdSeries | ndarray, ...]:
        """Apply `self.function` to `dataframe`.

        Args:
            dataframe: `pandas.DataFrame`. Dataframe to calculate the features on.
        Returns:
            calculated features.
        """
    @staticmethod
    def from_dict(data: dict[str, str | Callable | list[str]]):
        """Create a `CalculatedFeature` instance from a dict.

        Args:
            data: `dict`.
        Returns:
            CalculatedFeature instance.
        """
