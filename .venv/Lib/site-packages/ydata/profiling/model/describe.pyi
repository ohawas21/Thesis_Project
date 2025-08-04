import pandas as pd
from visions import VisionsTypeset as VisionsTypeset
from ydata.profiling.model.description import BaseDescription
from ydata_profiling.config import Settings
from ydata_profiling.model.summarizer import BaseSummarizer as BaseSummarizer

def describe(config: Settings, df: pd.DataFrame, summarizer: BaseSummarizer, typeset: VisionsTypeset, sample: dict | None = None) -> BaseDescription:
    """Calculate the statistics for each series in this DataFrame.

    Args:
        config: report Settings object
        df: DataFrame.
        summarizer: summarizer object
        typeset: visions typeset
        sample: optional, dict with custom sample

    Returns:
        This function returns a dictionary containing:
            - table: overall statistics.
            - variables: descriptions per series.
            - correlations: correlation matrices.
            - missing: missing value diagrams.
            - alerts: direct special attention to these patterns in your data.
            - package: package details.
    """
