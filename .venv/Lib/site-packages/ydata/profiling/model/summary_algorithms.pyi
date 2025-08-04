import pandas as pd
from scipy.stats import chisquare as chisquare
from ydata_profiling.config import Settings as Settings

def describe_categorical_1d(config: Settings, series: pd.Series, summary: dict) -> tuple[Settings, pd.Series, dict]: ...
