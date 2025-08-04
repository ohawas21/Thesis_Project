import abc
from abc import ABC
from dataclasses import dataclass
from numpy import ndarray as ndarray
from pandas import DataFrame as pdDataFrame
from typing import Generator as GeneratorType
from ydata.preprocessors.base import Preprocessor
from ydata.utils.random import RandomSeed as RandomSeed

@dataclass
class ConditionalFeature(ABC, metaclass=abc.ABCMeta):
    """Generic data class to represent a type of condition."""
    name: str
    def sample(self, n_samples: int, random_state: RandomSeed = None) -> ndarray:
        """Generic method to sample from the conditional feature."""
    def __init__(self, name) -> None: ...

@dataclass
class Category:
    """Generic class to represent a categorical value (i.e., category)."""
    VALUE_DTYPES = str | int | bool | float
    value: VALUE_DTYPES
    percentage: float = ...
    def __init__(self, value, percentage=...) -> None: ...

@dataclass
class CategoricalValues(ConditionalFeature):
    """Data class to represent a categorical conditional feature."""
    categories: list[Category | Category.VALUE_DTYPES | tuple[Category.VALUE_DTYPES, float] | dict] | None = ...
    balancing: bool = ...
    def __init__(self, name: str, categories: list[Category | Category.VALUE_DTYPES | tuple[Category.VALUE_DTYPES, float]] | None = None, balancing: bool = False, value_counts_normalized: dict = None) -> None: ...

@dataclass
class NumericalRange(ConditionalFeature):
    """Data class to represent a numerical conditional feature."""
    minimum: float
    maximum: float
    def __init__(self, name, minimum, maximum) -> None: ...

@dataclass
class Generator(ConditionalFeature):
    """Data class to represent a generic conditional feature based on a
    generator."""
    def __init__(self, name: str, function: GeneratorType) -> None: ...

class ConditionalFactory:
    """Factory used to generate the conditional objects."""
    @staticmethod
    def create_from_dict(condition_on: dict, data_types: dict, metadata_summary: dict) -> list[ConditionalFeature]:
        """Method to generate conditional objects from a dictionary."""

class ConditionalUtils:
    """Utility methods for the conditional objects."""
    @staticmethod
    def prepare_conditional_sample(condition_on: list[ConditionalFeature] | dict | pdDataFrame, conditional_features: list[str], data_types: dict, n_samples: int, preprocessor: Preprocessor, metadata_summary: dict, random_state: RandomSeed = None) -> pdDataFrame:
        """Method to prepare the conditional configuration for the sampling
        method.

        Returns the bootstrapping data.
        """
    @staticmethod
    def validate_conditional_features(condition_on: list[str], dataset_columns: list[str], anonymize_columns: list[str], dataset_attrs: dict):
        """Validate if the conditional features are admissible."""
    @staticmethod
    def validate_condition_types(condition_on: list[ConditionalFeature], data_types: dict):
        """Method to validate if the features match the condition types."""
