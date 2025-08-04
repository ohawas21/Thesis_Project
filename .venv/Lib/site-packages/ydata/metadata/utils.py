from collections.abc import Sequence
from numbers import Number
from typing import Any, Optional, Tuple, Union

import psutil
from dask.dataframe import DataFrame as ddDataFrame
from dask.dataframe import Series as ddSeries
from pandas import Series as pdSeries

from ydata.metadata.column import Column
from ydata.metadata.warning_types import Level, Warning, WarningType
from ydata.utils.data_types import _NULL_VALUES, CATEGORICAL_DTYPES, DataType

# Drop nulls logic extracted from dataprep https://github.com/sfu-db/dataprep/blob/develop/dataprep/eda/dtypes.py


# @typechecked
def drop_null(var: ddSeries, is_str: bool = False) -> ddSeries:
    var = var.dropna()
    if is_str:
        return var[~var.isin(_NULL_VALUES)]
    else:
        return var


# @typechecked
def _calc_warnings(warning_type: WarningType, value, column=None) -> Optional[Warning]:
    if warning_type == WarningType.SKEWNESS:
        warning, level = get_skewness_level(value)
    elif warning_type == WarningType.MISSINGS:
        warning, level = get_missingness_level(value)
    elif warning_type == WarningType.CARDINALITY:
        warning, level = get_cardinality_level(value)
    else:
        warning, level = get_duplicate_level(value)

    if warning:
        return Warning(
            warning_type=warning_type,
            column=column,
            details={"level": level, "value": value},
        )
    return None


# @typechecked
def _get_col_warnings(column: Column, summary: dict) -> dict:
    warnings = {}
    col = column.name
    if column.datatype == DataType.NUMERICAL:
        # Check wether the column has a warning concerning skewness
        skewness = summary["skewness"][col]
        warning = _calc_warnings(
            warning_type=WarningType.SKEWNESS, value=skewness, column=column
        )
        if warning:
            warnings[WarningType.SKEWNESS.value] = warning

    if column.datatype in [DataType.NUMERICAL, DataType.CATEGORICAL, DataType.STR, DataType.DATE]:
        # Check for column missing values level
        missingness = dict(summary["missings"])[col] / summary["nrows"]
        warning = _calc_warnings(
            warning_type=WarningType.MISSINGS, value=missingness, column=column
        )
        if warning:
            warnings[WarningType.MISSINGS.value] = warning

    if column.datatype in CATEGORICAL_DTYPES:
        cardinality = dict(summary["cardinality"])[col]
        warning = _calc_warnings(
            warning_type=WarningType.CARDINALITY, value=cardinality, column=column
        )
        if warning:
            warnings[WarningType.CARDINALITY.value] = warning

    return warnings


def get_skewness_level(skewness: Union[int, float]) -> Tuple:
    """
    Auxiliary method to calculate the skewness level based on the computed skewness.
    Rules:
    # less than -1 or greater than 1 highly skewed
    # Between -1 and -0.5 or between 0.5 and 1, moderately skewed
    # Between -0.5 and 0.5 is approx. symmetric
    """
    skewed = False
    level = None
    if skewness < -1 or skewness > 1:
        skewed = True
        level = Level.HIGH
    elif (skewness >= -1 and skewness <= -0.5) or (skewness >= 0.5 and skewness <= 1):
        skewed = True
        level = Level.MODERATE

    return skewed, level


def get_missingness_level(missingness: Union[int, float]) -> Tuple:
    """
    Auxiliary method to calculate the missinges level for a certain column
    Rules:
    # less than 30% -> considered to be normal
    # Between 30% and 60% moderate
    # Above 60% -> high levels of missingness
    """
    missing = False
    level = None
    if missingness > 0.3 and missingness <= 0.6:
        missing = True
        level = Level.MODERATE
    elif missingness > 0.6:
        missing = True
        level = Level.HIGH
    return (missing, level)


def get_cardinality_level(card_level: Union[int, float]) -> Tuple:
    """
    Auxiliary method to calculate the level of cardinality for a categorical variable
    Rules:
    # less than 30% -> considered to be normal
    # Between 30% and 60% moderate
    # Above 60% -> high levels of missingness
    """
    cardinality = False
    level = None
    if card_level > 20 and card_level < 50:
        cardinality = True
        level = Level.MODERATE
    elif card_level >= 50:
        cardinality = True
        level = Level.HIGH

    return (cardinality, level)


def get_duplicate_level(duplicates_ratio: float):
    """
    Auxiliary method that calculates the level of duplicates in the all dataset
    Rules:
    If duplicates between 10% and 30% level is considered to be moderate
    If duplicates ration above 30%, level is considered to be high
    """
    duplicates = False
    level = None
    if duplicates_ratio > 0.10 and duplicates_ratio <= 0.30:
        duplicates = True
        level = Level.MODERATE
    elif duplicates_ratio > 0.30:
        duplicates = True
        level = Level.HIGH

    return (duplicates, level)


def get_email_level(email_ratio: float):
    """Auxiliary method that calculated the probability level of a columns
    being an email.

    Rules:
    If regex matches>0.5 and matches <0.7 level is considered to be moderate
    If regex matches>=0.7 level is considered to be high
    """
    email = False
    level = None
    if email_ratio > 0.5 and email_ratio < 0.7:
        email = True
        level = Level.MODERATE
    elif email_ratio >= 0.7:
        email = True
        level = Level.HIGH
    return (email, level)


def value_counts(ddf: ddDataFrame, memory_ratio: int = 3) -> pdSeries:
    """Dynamically determine how to compute the value_counts."""
    table_size = ddf.memory_usage().sum().compute()
    memory_usage = psutil.virtual_memory()
    if memory_usage.available / table_size > memory_ratio:
        return ddf.compute().value_counts()
    hash_values = ddf.astype('str').apply(
        lambda x: '_'.join(x.values), axis=1, meta=(None, 'str'))
    result = hash_values.value_counts()
    return result.compute()


def dropna(ddf: ddDataFrame, how: str = "all", memory_ratio: int = 3) -> pdSeries:
    """Dynamically determine how to compute the missing values."""
    table_size = ddf.memory_usage().sum().compute()
    memory_usage = psutil.virtual_memory()
    if memory_usage.available / table_size > memory_ratio:
        return ddf.compute().dropna(how="all")
    return ddf.dropna(how="all").compute()


def is_sequence(value: Any) -> bool:
    """Check if value is a non-string sequence.

    Args:
        value (Any): any value

    Returns:
        bool: True if is a non-string sequence and False otherwise
    """
    if isinstance(value, str):
        return False
    return isinstance(value, Sequence)


def is_non_negative_number(value: Any) -> bool:
    """Check if value is a non-negative number.

    Args:
        value (Any): any value

    Returns:
        bool: True if value is a non-negative number and False otherwise.
    """
    return isinstance(value, Number) and value >= 0
