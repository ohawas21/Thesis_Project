from ydata.report.metrics.base_metric import BaseMetric

class StatisticalSimilarity(BaseMetric):
    """Statistical similarity metric.

    This metric is only computed for numerical variables.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...
