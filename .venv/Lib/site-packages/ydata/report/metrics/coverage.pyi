import pandas as pd
from ydata.report.metrics.top_bottom_metric import TopBottomMetric

class CategoryCoverage(TopBottomMetric):
    """Category Coverage Metric.

    Compute the fraction of real data categories that are also
    represented in the synthetic data. Provide the ratio of
    representation of the category in the real and in the synthetic
    dataset.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...

class MissingCategoryCoverage(TopBottomMetric):
    """Missing Category Coverage Metric.

    Compute the categorical values from the real data that are not
    represented in the synthetic data. Provide the categories that have
    not been represented.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...

class RangeCoverage(TopBottomMetric):
    """Range Coverage Metric.

    Compute the numerical variables ranges found in the real dataset are
    also represented in the synthetic.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...
    def get_minmax(self, dataset: pd.DataFrame):
        """Calculates the min-max stats of a provided dataframe."""
