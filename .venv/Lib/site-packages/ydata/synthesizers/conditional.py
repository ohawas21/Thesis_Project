from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import Generator as GeneratorType

from numpy import array
from numpy import isnan as npisnan
from numpy import nan as npnan
from numpy import ndarray
from numpy.random import default_rng
from pandas import DataFrame as pdDataFrame

from ydata.dataset import Dataset
from ydata.preprocessors.base import Preprocessor
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed


@dataclass
class ConditionalFeature(ABC):
    """Generic data class to represent a type of condition."""
    name: str
    """Name of the feature to which the condition is applied."""

    def sample(self, n_samples: int, random_state: RandomSeed = None) -> ndarray:
        """Generic method to sample from the conditional feature."""
        return self._sample(n_samples=n_samples, random_state=random_state)

    @abstractmethod
    def _sample(self, n_samples: int, random_state: RandomSeed = None) -> ndarray:
        """Generic method to generate the data of the conditional feature."""
        pass


@dataclass
class Category:
    """Generic class to represent a categorical value (i.e., category)."""
    VALUE_DTYPES = str | int | bool | float
    """The categorical value possible types."""
    value: VALUE_DTYPES
    """The categorical value (i.e., category)."""
    percentage: float = npnan
    """The percentage of rows the category will take on the synthetic dataset."""


@dataclass
class CategoricalValues(ConditionalFeature):
    """Data class to represent a categorical conditional feature."""
    categories: list[Category |
                     Category.VALUE_DTYPES |
                     tuple[Category.VALUE_DTYPES, float] | dict] | None = None
    """List of categories to condition upon."""
    balancing: bool = False
    """If true all categories have the same weight (the list of categories is ignored)."""

    def __init__(self, name: str,
                 categories: list[Category | Category.VALUE_DTYPES |
                                  tuple[Category.VALUE_DTYPES, float]] | None = None,
                 balancing: bool = False,
                 value_counts_normalized: dict = None):
        super().__init__(name=name)
        self.balancing = balancing

        if self.balancing:
            original_categories = list(value_counts_normalized[name].keys())
            perc = 1.0 / len(original_categories)
            self.categories = [Category(value=cat, percentage=perc)
                               for cat in original_categories]
        else:
            if not categories or len(categories) == 0:
                raise ValueError("No categories were supplied.")

            self.categories = []
            for cat in categories:
                if isinstance(cat, Category.VALUE_DTYPES):
                    self.categories.append(
                        Category(value=cat, percentage=npnan))
                elif isinstance(cat, tuple) and self._is_tuple_valid(cat):
                    self.categories.append(
                        Category(value=cat[0], percentage=float(cat[1])))
                elif isinstance(cat, dict) and self._is_dict_valid(cat):
                    perc = float(cat["percentage"]
                                 ) if "percentage" in cat else npnan
                    self.categories.append(
                        Category(value=cat["category"], percentage=perc))
                elif isinstance(cat, Category):
                    self.categories.append(cat)
                else:
                    raise TypeError("Invalid category.")

            if value_counts_normalized is not None:
                provided_categories = [c.value for c in self.categories]
                not_provided_categories = [Category(value=k, percentage=npnan)
                                           for k in list(value_counts_normalized[name].index)
                                           if k not in provided_categories]
                self.categories.extend(not_provided_categories)

                self.categories = self._adjust_percentages(
                    self.name, self.categories, value_counts_normalized)

        self._validate_percentages(
            list(map(lambda cv: cv.percentage, self.categories)))

    @staticmethod
    def _adjust_percentages(name: str, categories: list[Category], value_counts_normalized: dict):
        supplied_categories = [
            cv for cv in categories if not npisnan(cv.percentage)]
        non_supplied_categories = [
            cv for cv in categories if npisnan(cv.percentage)]
        supplied_values = list(
            map(lambda cv: cv.value, supplied_categories))
        supplied_percentage_sum = sum(
            list(map(lambda cv: cv.percentage, supplied_categories)))
        if supplied_percentage_sum > 1.0:
            raise ValueError("The sum of the percentages supplied for the values of " +
                             "a conditional categorical feature cannot be greater than 1.")
        free_percentage_sum = 1.0 - \
            sum([value_counts_normalized[name][v]
                for v in supplied_values])
        adjustment_per_category = (free_percentage_sum -
                                   sum([value_counts_normalized[name][cv.value]
                                        for cv in non_supplied_categories])) / len(non_supplied_categories)
        for cat in categories:
            if npisnan(cat.percentage):
                cat.percentage = ((1.0 - supplied_percentage_sum) *
                                  (value_counts_normalized[name][cat.value] + adjustment_per_category)) / free_percentage_sum
        return categories

    @staticmethod
    def _validate_percentages(percentages: list[float]):
        if not all(p >= 0.0 and p <= 1.0 for p in percentages) or round(sum(percentages), 2) != 1.0:
            raise ValueError(
                "Invalid percentages supplied for the values of the conditional categorical feature.")

    @staticmethod
    def _is_tuple_valid(category: tuple[Category.VALUE_DTYPES, float]) -> bool:
        if len(category) < 2:
            raise ValueError(
                "The tuple must contain the value/category and the percentage.")
        if not isinstance(category[0], Category.VALUE_DTYPES):
            raise ValueError("Invalid value/category.")
        try:
            float(category[1])
        except ValueError:
            raise ValueError("Invalid percentage.")
        return True

    @staticmethod
    def _is_dict_valid(category: dict) -> bool:
        if len(category) == 0:
            raise ValueError("The dictionary cannot be empty.")
        if "category" not in category:
            raise ValueError("Key 'category' is missing.")
        if not isinstance(category["category"], Category.VALUE_DTYPES):
            raise ValueError("Invalid value/category.")
        if "percentage" in category:
            try:
                float(category["percentage"])
            except ValueError:
                raise ValueError("Invalid percentage.")
        return True

    def _sample(self, n_samples: int, random_state: RandomSeed = None) -> ndarray:
        """Method to generate the data of the categorical conditional
        feature."""
        rng = default_rng(seed=random_state)
        percentages = list(map(lambda cv: cv.percentage, self.categories))
        self._validate_percentages(percentages)
        values = list(map(lambda cv: cv.value, self.categories))
        return rng.choice(values, size=n_samples, p=percentages, replace=True)


@dataclass
class NumericalRange(ConditionalFeature):
    """Data class to represent a numerical conditional feature."""
    minimum: float
    """Minimum of the range to condition upon."""
    maximum: float
    """Maximum of the range to condition upon."""

    def _sample(self, n_samples: int, random_state: RandomSeed = None) -> ndarray:
        """Method to generate the data of the numerical conditional feature."""
        rng = default_rng(seed=random_state)
        if self.maximum < self.minimum:
            raise ValueError(
                "The supplied range for the conditional numerical feature is invalid.")
        return rng.uniform(low=self.minimum, high=self.maximum, size=n_samples)


@dataclass
class Generator(ConditionalFeature):
    """Data class to represent a generic conditional feature based on a
    generator."""
    _instance: GeneratorType
    """Instance of the generator."""

    def __init__(self, name: str, function: GeneratorType):
        super().__init__(name=name)
        self._instance = function()

    def _sample(self, n_samples: int, random_state: RandomSeed = None) -> ndarray:
        """Method to generate the data of the generic conditional feature."""
        data = array(list(islice(self._instance, n_samples)))
        if len(data) < n_samples:
            raise ValueError(
                "The generator is unable to produce enough values for the sampling procedure.")
        return data


class ConditionalFactory:
    """Factory used to generate the conditional objects."""

    @staticmethod
    def create_from_dict(condition_on: dict, data_types: dict, metadata_summary: dict) -> list[ConditionalFeature]:
        """Method to generate conditional objects from a dictionary."""

        value_counts_normalized = {
            key: value / metadata_summary["nrows"] for key, value in metadata_summary["value_counts"].items()}
        mapped_conditions = []
        for col_name, cond_params in condition_on.items():
            if isinstance(cond_params, dict):
                if "categories" in cond_params or "balancing" in cond_params:
                    balancing = cond_params.get("balancing", False)
                    categories = cond_params.get("categories", [])
                    mapped_conditions.append(CategoricalValues(name=col_name,
                                                               balancing=balancing,
                                                               categories=categories,
                                                               value_counts_normalized=value_counts_normalized))
                elif "minimum" in cond_params and "maximum" in cond_params:
                    mapped_conditions.append(NumericalRange(name=col_name,
                                                            minimum=cond_params["minimum"],
                                                            maximum=cond_params["maximum"]))
                elif "function" in cond_params:
                    mapped_conditions.append(Generator(name=col_name,
                                                       function=cond_params["function"]))
            else:
                if data_types[col_name] in [DataType.CATEGORICAL, DataType.STR]:
                    if not isinstance(cond_params, list):
                        cond_params = [cond_params]
                    mapped_conditions.append(CategoricalValues(name=col_name,
                                                               categories=cond_params,
                                                               value_counts_normalized=value_counts_normalized))
                elif data_types[col_name] == DataType.NUMERICAL:
                    mapped_conditions.append(NumericalRange(name=col_name,
                                                            minimum=cond_params[0],
                                                            maximum=cond_params[1]))
                else:
                    raise ValueError(
                        f"Invalid conditional parameters for feature '{col_name}'.")
        return mapped_conditions


class ConditionalUtils:
    """Utility methods for the conditional objects."""

    @staticmethod
    def _convert_processed_dtypes(df: pdDataFrame, data_types: dict):
        for col in df.columns:
            isna = df[col].isna()
            if isna.any() and data_types[col] == DataType.NUMERICAL:
                df[col] = df[col].astype('float')

    @staticmethod
    def prepare_conditional_sample(condition_on: list[ConditionalFeature] | dict | pdDataFrame,
                                   conditional_features: list[str],
                                   data_types: dict,
                                   n_samples: int,
                                   preprocessor: Preprocessor,
                                   metadata_summary: dict,
                                   random_state: RandomSeed = None) -> pdDataFrame:
        """Method to prepare the conditional configuration for the sampling
        method.

        Returns the bootstrapping data.
        """
        if isinstance(condition_on, dict):
            condition_on = ConditionalFactory.create_from_dict(
                condition_on, data_types, metadata_summary)

        # REMOVED DURING THE NEW MULTI-TABLE DEVELOPMENTS
        # cond_cols = list(map(lambda cf: cf.name, condition_on))
        # if set(conditional_features) != set(cond_cols):
        #    raise ValueError(
        #        "The conditional features do not match the list supplied to the fit method.")

        if isinstance(condition_on, pdDataFrame):
            bootstrapping_data = condition_on.copy()
        else:
            ConditionalUtils.validate_condition_types(condition_on=condition_on,
                                                      data_types=data_types)
            bootstrapping_data = {}
            for cf in condition_on:
                bootstrapping_data[cf.name] = cf.sample(
                    n_samples=int(n_samples), random_state=random_state)
            bootstrapping_data = pdDataFrame.from_dict(bootstrapping_data)

        if preprocessor is not None:
            bootstrapping_data = preprocessor.transform(
                Dataset(bootstrapping_data)).to_pandas()
            ConditionalUtils._convert_processed_dtypes(
                bootstrapping_data,
                data_types,
            )

        # The integer values must be converted from int8 to int64.
        for col in bootstrapping_data.columns:
            if bootstrapping_data.dtypes[col] == "int8":
                bootstrapping_data[col] = bootstrapping_data[col].astype(
                    'int64')

        return bootstrapping_data

    @staticmethod
    def validate_conditional_features(condition_on: list[str],
                                      dataset_columns: list[str],
                                      anonymize_columns: list[str],
                                      dataset_attrs: dict):
        """Validate if the conditional features are admissible."""
        if not all(col in dataset_columns for col in condition_on):
            raise ValueError(
                "Some of the conditional features are not part of the dataset.")

        if anonymize_columns is not None and any(col in anonymize_columns for col in condition_on):
            raise ValueError("Conditional features cannot be anonymized.")

        if dataset_attrs is not None:
            sortbykey = dataset_attrs.sortbykey \
                if isinstance(dataset_attrs.sortbykey, list) \
                else [dataset_attrs.sortbykey]
            if any(col in sortbykey for col in condition_on):
                raise ValueError(
                    f"The sorting features (sortbykey={sortbykey}) cannot be conditioned upon.")

            entities = dataset_attrs.entities \
                if isinstance(dataset_attrs.entities, list) \
                else [dataset_attrs.entities]
            if any(col in entities for col in condition_on):
                raise ValueError(
                    f"The entity features (entities={entities}) cannot be conditioned upon.")

    @staticmethod
    def validate_condition_types(condition_on: list[ConditionalFeature],
                                 data_types: dict):
        """Method to validate if the features match the condition types."""
        for cf in condition_on:
            if isinstance(cf, CategoricalValues) and data_types[cf.name] not in [DataType.CATEGORICAL, DataType.STR]:
                raise TypeError(
                    f"Feature '{cf.name}' can not be conditioned as categorical.")
            elif isinstance(cf, NumericalRange) and data_types[cf.name] != DataType.NUMERICAL:
                raise TypeError(
                    f"Feature '{cf.name}' can not be conditioned as numerical.")
