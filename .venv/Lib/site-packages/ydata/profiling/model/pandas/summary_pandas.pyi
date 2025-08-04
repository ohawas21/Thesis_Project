import pandas as pd
from tqdm import tqdm as tqdm
from visions import VisionsTypeset as VisionsTypeset
from ydata_profiling.config import Settings as Settings
from ydata_profiling.model.summarizer import BaseSummarizer as BaseSummarizer

def pandas_describe_1d(config: Settings, series: pd.Series, summarizer: BaseSummarizer, typeset: VisionsTypeset) -> dict:
    """Describe a series (infer the variable type, then calculate type-specific values).

    Args:
        config: report Settings object
        series: The Series to describe.
        summarizer: Summarizer object
        typeset: Typeset

    Returns:
        A Series containing calculated series description values.
    """
def pandas_get_series_descriptions(config: Settings, df: pd.DataFrame, summarizer: BaseSummarizer, typeset: VisionsTypeset, pbar: tqdm) -> dict: ...
