from ydata.report.metrics.top_bottom_metric import TopBottomMetric

class KSTest(TopBottomMetric):
    """Kolmogorov-Smirnov test based metric for numerical columns.

    This metric uses the Kolmogorov-Smirnov test to compare the
    distribution between two continuous columns (real and synth) using
    the empirical CDF. The test returns the resulting p-value. A small
    value means that we can reject the null hypothesis, meaning, the
    distributions are most probably different.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...

class TotalVariationDistance(TopBottomMetric):
    """Total Variation Distance metric for categorical columns.

    This metric computes the Total Variation Distance to compare the
    marginal distribution between two categorical columns (real and
    synth). The metric returns the complement of the Total Variation
    Distance so that higher values represent more similar distributions.
    """
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...
