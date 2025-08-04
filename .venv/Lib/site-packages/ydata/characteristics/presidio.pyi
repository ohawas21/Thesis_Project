from pandas import Series as pdSeries

def detect_characteristics(columns: dict[str, pdSeries], threshold: float) -> dict:
    """Detect the characteristics using presidio.

    Args:
        columns (dict[str, pdSeries]): dictionary of pandas Series (each series might have a different size)
        threshold (float): threshold used to filter entities with Presidio scores below this value

    Return:
        dictionary indexed on the column, with a dictionary of characteristics: values where value is in [0, 1]
    """
