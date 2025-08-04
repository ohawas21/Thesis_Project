import pandas as pd
from ydata_profiling.config import Settings

def pandas_auto_compute(config: Settings, df: pd.DataFrame, summary: dict) -> pd.DataFrame | None: ...
