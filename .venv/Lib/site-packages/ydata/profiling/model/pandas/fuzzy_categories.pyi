import pandas as pd

def get_fuzzy_categories(series: pd.Series, threshold: float = 0.6):
    """
    Identifies fuzzy/dirty categories efficiently and calculates risk based on fuzzy ratio.
    Ensures numeric variations are ignored, and unique values are counted post-normalization.

    Args:
        series: a pandas series with the column values
        threshold: the fuzzy ratio to identify potential fuzziness
        risk_threshold: defines the level of risk that it might bring to the data quality

    Returns: the ratio of fuzziness identified as well as whether poses a risk
    """
