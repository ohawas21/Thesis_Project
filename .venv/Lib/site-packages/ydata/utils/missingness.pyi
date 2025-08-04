from pandas import DataFrame as DataFrame

def calculate_missing_percentages(data: DataFrame) -> dict:
    """
    Calculates the percentages of the missingness in the columns.
    Args:
        data: pd.DataFrame. The dataset we are analyzing.

    Returns: dict[str, int]. Dictionary containing the percentage of missingness per column.
    """
