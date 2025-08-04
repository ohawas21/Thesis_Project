from typing import Any, Dict

from pandas import DataFrame as pdDataFrame
from ydata_profiling.config import Settings
from ydata_profiling.report.presentation.core import Image
from ydata_profiling.visualisation.utils import plot_360_n0sc0pe

from ydata.dataset.dataset import Dataset
from ydata.metadata.metadata import Metadata
from ydata.profiling.logger import profilinglogger_config
from ydata.quality.outlier import OutlierEngine


def describe_outliers(dataframe: pdDataFrame, config: Settings) -> Dict[str, Any]:
    """Computes outliers and generate plots.

    Args:
        dataframe (pdDataFrame): input data
        config (Settings): report config

    Returns:
        Dict[str, Any]: dictionary containing outliers and plots
    """
    logger = profilinglogger_config(verbose=False)
    data = Dataset(dataframe)
    meta = Metadata(data)
    # skip the outliers if there are too few numerical vars
    if len(meta.numerical_vars) < 2:
        return {}
    try:
        oe = OutlierEngine()
        outliers = oe.fit_predict(data, meta)
        summary = oe.summary()
        plots = [{"name": name, "image": _plot_pipeline(data, pipe, config)}
                 for name, pipe in oe.pipelines.items()]
        return {
            "outliers": outliers,
            "summary": summary,
            "plots": plots,
        }
    except Exception as e:
        logger.error(e)

    return {}


def _plot_pipeline(data: Dataset, pipeline, config: Settings) -> Image:
    pipeline.plot(data)
    return plot_360_n0sc0pe(config)
