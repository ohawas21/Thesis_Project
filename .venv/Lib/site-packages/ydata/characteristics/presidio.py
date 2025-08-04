import logging

from pandas import Series as pdSeries
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine

from ydata.characteristics import ColumnCharacteristic

logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


def detect_characteristics(columns: dict[str, pdSeries], threshold: float) -> dict:
    """Detect the characteristics using presidio.

    Args:
        columns (dict[str, pdSeries]): dictionary of pandas Series (each series might have a different size)
        threshold (float): threshold used to filter entities with Presidio scores below this value

    Return:
        dictionary indexed on the column, with a dictionary of characteristics: values where value is in [0, 1]
    """
    analyzer = AnalyzerEngine()
    PRESIDIO_TO_CHARAC = {
        "EMAIL_ADDRESS": ColumnCharacteristic.EMAIL,
        "CREDIT_CARD": ColumnCharacteristic.CREDIT_CARD,
        "PERSON": ColumnCharacteristic.PERSON,
        "LOCATION": ColumnCharacteristic.LOCATION,
        "URL": ColumnCharacteristic.URL
    }
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    # Pre-process the data from each column.
    char_presidio = {col: list(df.values) for col, df in columns.items()}

    # Run Presidio to recognize entities in each column.
    char_presidio = {col: list(batch_analyzer.analyze_dict({col: df}, language="en", entities=list(
        PRESIDIO_TO_CHARAC.keys()))) for col, df in char_presidio.items()}

    # Calculates the percentage of rows from the sample that have entities recognized.
    char_presidio = {col: {c.key: pdSeries(
        # Entities recognized with a Presidio score above the specified threshold.
        [e[0].entity_type for e in c.recognizer_results if e and e[0].score >= threshold] +
        # Entities recognized with a Presidio score below the specified threshold.
        ["BELOW_THRESHOLD_" + e[0].entity_type for e in c.recognizer_results if e and e[0].score < threshold] +
        # Values without recognized entities.
        ["NONE" for e in c.recognizer_results if not e],
        dtype=str).value_counts(normalize=True) for c in results} for col, results in char_presidio.items()}

    # Transforms the dictionary to the output format, while only keeping the recognized entities.
    char_presidio = {k: {k2: v2 for k2, v2 in dict(zip(list(map(PRESIDIO_TO_CHARAC.get, v.index)), v.values)).items() if k2 is not None}
                     for results in char_presidio.values() for k, v in results.items()}

    # Removes from the dictionary columns without recognized entities.
    char_presidio = {k: v for k, v in char_presidio.items() if v}
    return char_presidio
