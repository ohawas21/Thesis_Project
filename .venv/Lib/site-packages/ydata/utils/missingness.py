from pandas import DataFrame


def calculate_missing_percentages(data: DataFrame) -> dict:
    """
    Calculates the percentages of the missingness in the columns.
    Args:
        data: pd.DataFrame. The dataset we are analyzing.

    Returns: dict[str, int]. Dictionary containing the percentage of missingness per column.
    """
    return round(data.isna().sum() / len(data) * 100, 2).to_dict()
