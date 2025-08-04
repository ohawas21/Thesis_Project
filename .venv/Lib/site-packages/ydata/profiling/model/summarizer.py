from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from visions import VisionsTypeset

from ydata_profiling.config import Settings
from ydata_profiling.model import BaseDescription

from ydata_profiling.model.pandas.describe_supported_pandas import (
    pandas_describe_supported,
)

from ydata_profiling.model.pandas import (
    pandas_describe_boolean_1d,
    pandas_describe_counts,
    pandas_describe_date_1d,
    pandas_describe_file_1d,
    pandas_describe_generic,
    pandas_describe_image_1d,
    pandas_describe_numeric_1d,
    pandas_describe_path_1d,
    pandas_describe_text_1d,
    pandas_describe_timeseries_1d,
    pandas_describe_url_1d,
)
from ydata.profiling.model.pandas.describe_categorical_pandas import pandas_describe_categorical_1d

from ydata_profiling.model.summarizer import ProfilingSummarizer

class YDataProfilingSummarizer(ProfilingSummarizer):
    """The default YData Profiling summarizer"""

    #Update here the summary map? Will this work?
    def _create_summary_map(self) -> Dict[str, List[Callable]]:
        """Creates the summary map for Pandas summarization."""
        summary_map = {
            "Unsupported": [
                pandas_describe_counts,
                pandas_describe_generic,
                pandas_describe_supported,
            ],
            "Numeric": [pandas_describe_numeric_1d],
            "DateTime": [pandas_describe_date_1d],
            "Text": [pandas_describe_text_1d],
            "Categorical": [pandas_describe_categorical_1d],
            "Boolean": [pandas_describe_boolean_1d],
            "URL": [pandas_describe_url_1d],
            "Path": [pandas_describe_path_1d],
            "File": [pandas_describe_file_1d],
            "Image": [pandas_describe_image_1d],
            "TimeSeries": [pandas_describe_timeseries_1d],
        }
        return summary_map

