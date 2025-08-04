"""
    File to define calculated features specific methods
"""
from typing import Callable

from numpy import ndarray
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries

from ydata.dataset.dataset import Dataset

from ydata.synthesizers.exceptions import SynthesizerAssertionError

def _validate_calculated_features(calculated_features: list,
                                   dataset: Dataset) -> None:
    """
    Validates the configuration of calculated features before adding them to the dataset.

    Args:
        calculated_features (list): A list of dictionaries, where each dictionary defines a calculated feature.
        dataset (Dataset): The dataset to which the calculated features will be added.

    Raises:
        AssertionError: If any calculated feature definition is invalid or incompatible with the dataset.
    """
    for c in calculated_features:
        func = c['function']
        origin = c["calculated_from"]

        if not all(e in list(dataset.columns) for e in origin):
            raise SynthesizerAssertionError('Ensure that the specified origin columns exist in the provided dataset. '
                                 'Calculated features can only be derived from existing columns.')

        if not isinstance(func, Callable):
            raise SynthesizerAssertionError('Ensure that your provide a valid function for each calculated feature. '
                                 'A function must be a Python Callable object.')

#TODO revisit along with the MultiTable refactor
class CalculatedFeature:
    def __init__(
        self,
        features: str | list[str],
        function: Callable,
        calculated_from: list[str] | str,
        reference_keys: dict[str, list | str] | None = None,
    ) -> None:
        self._features = features
        self._function = function
        self._calculated_from = calculated_from
        self._source_table_keys = None
        self._feature_table_keys = None
        if reference_keys is not None:
            assert "source" in reference_keys
            assert "target" in reference_keys
            if isinstance(reference_keys["source"], str):
                self._source_table_keys = [reference_keys["source"]]
            else:
                self._source_table_keys = reference_keys["source"]
            if isinstance(reference_keys["target"], str):
                self._source_table_keys = [reference_keys["target"]]
            else:
                self._feature_table_keys = reference_keys["target"]
            assert len(self._feature_table_keys) == len(
                self._source_table_keys)

    @property
    def features(self) -> list[str]:
        if type(self._features) == str:
            return [self._features]
        return self._features

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def calculated_from(self) -> list[str]:
        if type(self._calculated_from) == str:
            return [self._calculated_from]
        return self._calculated_from

    @property
    def reference_keys(self) -> list[tuple[str]]:
        if self._source_table_keys is None:
            return []
        return list(zip(self._feature_table_keys, self._source_table_keys))

    def apply_to(
        self, dataframe: pdDataFrame
    ) -> pdSeries | ndarray | tuple[pdSeries | ndarray, ...]:
        """Apply `self.function` to `dataframe`.

        Args:
            dataframe: `pandas.DataFrame`. Dataframe to calculate the features on.
        Returns:
            calculated features.
        """
        return self.function(*[dataframe[c] for c in self.calculated_from])

    @staticmethod
    def from_dict(data: dict[str, str | Callable | list[str]]):
        """Create a `CalculatedFeature` instance from a dict.

        Args:
            data: `dict`.
        Returns:
            CalculatedFeature instance.
        """
        CalculatedFeature.__validate_dict(data)
        return CalculatedFeature(
            features=data["calculated_features"],
            function=data["function"],
            calculated_from=data["calculated_from"],
            reference_keys=data.get("reference_keys", None),
        )

    @staticmethod
    def __validate_dict(data):
        assert (
            data is not None and len(data) > 0
        ), "Cannot create a caculated feature from a empty dict."
        assert "calculated_features" in data, "calculated_features not informed."
        assert "function" in data, "function not informed."
        assert "calculated_from" in data, "calculated_from not informed."

    def __str__(self) -> str:
        string = f"CalculatedFeature(features={self.features}, "
        string += f"function={self.function}, "
        string += f"calculated_from={self.calculated_from}"
        if self.reference_keys:
            string += f", reference_keys={self.reference_keys}"
        string += ")"
        return string

    def __repr__(self) -> str:
        return str(self)


def __validate_calculated_features(calculated_features: list, dataset: Dataset) -> None:
    return None
