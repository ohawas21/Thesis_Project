import pandas as pd
from collections import Counter
from ydata.profiling.model.summary_algorithms import describe_categorical_1d as describe_categorical_1d
from ydata_profiling.config import Settings as Settings

def get_character_counts_vc(vc: pd.Series) -> pd.Series: ...
def get_character_counts(series: pd.Series) -> Counter:
    """Function to return the character counts

    Args:
        series: the Series to process

    Returns:
        A dict with character counts
    """
def counter_to_series(counter: Counter) -> pd.Series: ...
def unicode_summary_vc(vc: pd.Series) -> dict: ...
def word_summary_vc(vc: pd.Series, stop_words: list[str] = []) -> dict:
    """Count the number of occurrences of each individual word across
    all lines of the data Series, then sort from the word with the most
    occurrences to the word with the least occurrences. If a list of
    stop words is given, they will be ignored.

    Args:
        vc: Series containing all unique categories as index and their
            frequency as value. Sorted from the most frequent down.
        stop_words: List of stop words to ignore, empty by default.

    Returns:
        A dict containing the results as a Series with unique words as
        index and the computed frequency as value
    """
def length_summary_vc(vc: pd.Series) -> dict: ...
def pandas_describe_categorical_1d(config: Settings, series: pd.Series, summary: dict) -> tuple[Settings, pd.Series, dict]:
    """Describe a categorical series.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
