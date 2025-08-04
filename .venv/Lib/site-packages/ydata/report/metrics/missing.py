import pandas as pd

from ydata.report.metrics.top_bottom_metric import TopBottomMetric
from ydata.report.styles.html import StyleHTML


class MissingValuesSimilarity(TopBottomMetric):
    """Missing values similarity metric.

    This metric calculates the similarity between the percentages of
    missing values in the real and synthetic data.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @property
    def name(self) -> str:
        return "Missing Values Similarity"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('MISSING VALUES SIMILARITY (MVS)')} measures how close \
            are the percentages of missing values in the synthetic and real data. This metric \
            is bounded between [0-1], where 1 represents the same percentage of missing data. \
            The two tables below present the five features with the highest and lowest similarity."

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the missing values similarity metric.
        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.

        Returns: the missing values similarity metric.
        """
        stats_mvs = kwargs["stats_summary"]["missing_values"]
        results = {}
        for col in stats_mvs["real"].keys():
            real_mr = stats_mvs["real"][col] / source.shape[0]
            synth_mr = stats_mvs["synth"][col] / synthetic.shape[0]
            results[col] = 1 - abs(real_mr - synth_mr)
        return self._get_results(results=results, name="MVS")
