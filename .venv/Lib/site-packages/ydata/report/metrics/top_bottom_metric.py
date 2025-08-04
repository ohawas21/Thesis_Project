import numpy as np
import pandas as pd

from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.styles.html import StyleHTML


class TopBottomMetric(BaseMetric):
    """Generic class for metrics that return top and bottom records."""

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)
        self._NUMBER_COLUMNS_TO_KEEP = 5

    def _get_mean_result(self, results: dict):
        return np.mean(list(results.values()))

    def _get_top_and_bottom_results(self, results: dict, name: str):
        sorted_cols = sorted(results, key=results.get, reverse=True)
        top_k_cols = sorted_cols[:self._NUMBER_COLUMNS_TO_KEEP]
        bottom_k_cols = sorted_cols[-self._NUMBER_COLUMNS_TO_KEEP:]

        results_top_k_cols = pd.DataFrame.from_dict({
            "Feature": top_k_cols,
            f"{name} (Highest)": [results.get(k) for k in top_k_cols]
        }).round(2)

        results_bottom_k_cols = pd.DataFrame.from_dict({
            "Feature": bottom_k_cols,
            f"{name} (Lowest)": [results.get(k) for k in bottom_k_cols]
        }).round(2)

        return {
            "table_top_k_cols": results_top_k_cols,
            "table_bottom_k_cols": results_bottom_k_cols
        }

    def _get_results(self, results: dict, name: str):
        return {
            "mean": self._get_mean_result(results=results),
            "dataframes": self._get_top_and_bottom_results(results=results, name=name)
        }
