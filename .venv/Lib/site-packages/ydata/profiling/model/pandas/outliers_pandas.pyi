from pandas import DataFrame as pdDataFrame
from typing import Any
from ydata_profiling.config import Settings as Settings
from ydata_profiling.report.presentation.core import Image as Image

def describe_outliers(dataframe: pdDataFrame, config: Settings) -> dict[str, Any]:
    """Computes outliers and generate plots.

    Args:
        dataframe (pdDataFrame): input data
        config (Settings): report config

    Returns:
        Dict[str, Any]: dictionary containing outliers and plots
    """
