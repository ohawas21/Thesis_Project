from ydata.report.metrics.top_bottom_metric import TopBottomMetric

class MissingValuesSimilarity(TopBottomMetric):
    """Missing values similarity metric.

    This metric calculates the similarity between the percentages of
    missing values in the real and synthetic data.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...
