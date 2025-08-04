from dask.dataframe import DataFrame as ddDataFrame, Series as ddSeries
from pandas import Series as pdSeries
from typing import Any

def drop_null(var: ddSeries, is_str: bool = False) -> ddSeries: ...
def get_skewness_level(skewness: int | float) -> tuple:
    """
    Auxiliary method to calculate the skewness level based on the computed skewness.
    Rules:
    # less than -1 or greater than 1 highly skewed
    # Between -1 and -0.5 or between 0.5 and 1, moderately skewed
    # Between -0.5 and 0.5 is approx. symmetric
    """
def get_missingness_level(missingness: int | float) -> tuple:
    """
    Auxiliary method to calculate the missinges level for a certain column
    Rules:
    # less than 30% -> considered to be normal
    # Between 30% and 60% moderate
    # Above 60% -> high levels of missingness
    """
def get_cardinality_level(card_level: int | float) -> tuple:
    """
    Auxiliary method to calculate the level of cardinality for a categorical variable
    Rules:
    # less than 30% -> considered to be normal
    # Between 30% and 60% moderate
    # Above 60% -> high levels of missingness
    """
def get_duplicate_level(duplicates_ratio: float):
    """
    Auxiliary method that calculates the level of duplicates in the all dataset
    Rules:
    If duplicates between 10% and 30% level is considered to be moderate
    If duplicates ration above 30%, level is considered to be high
    """
def get_email_level(email_ratio: float):
    """Auxiliary method that calculated the probability level of a columns
    being an email.

    Rules:
    If regex matches>0.5 and matches <0.7 level is considered to be moderate
    If regex matches>=0.7 level is considered to be high
    """
def value_counts(ddf: ddDataFrame, memory_ratio: int = 3) -> pdSeries:
    """Dynamically determine how to compute the value_counts."""
def dropna(ddf: ddDataFrame, how: str = 'all', memory_ratio: int = 3) -> pdSeries:
    """Dynamically determine how to compute the missing values."""
def is_sequence(value: Any) -> bool:
    """Check if value is a non-string sequence.

    Args:
        value (Any): any value

    Returns:
        bool: True if is a non-string sequence and False otherwise
    """
def is_non_negative_number(value: Any) -> bool:
    """Check if value is a non-negative number.

    Args:
        value (Any): any value

    Returns:
        bool: True if value is a non-negative number and False otherwise.
    """
