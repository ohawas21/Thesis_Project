from ydata.report.metrics.base_metric import BaseMetric

class DistanceCorrelation(BaseMetric):
    """Calculate the correlation/strength-of-association of features in data-
    set with both categorical and continuous.

    features using: * Pearson's R for continuous-continuous cases * Correlation Ratio for categorical-continuous cases
    * Cramer's V or Theil's U for categorical-categorical cases.
    The function then computes the correlation score, which is the average of the difference between the elements of the
    the correlation matrices. The result will be a score between 0 and 1, with a score of 1 declaring that the
    correlation matrices are identical.
    """
    def __init__(self, formatter=...) -> None: ...
    def distance_correlation(self, source, synthetic, metadata): ...
    @property
    def name(self) -> str: ...

def distance_distribution(df_real, df_synth, data_types: dict):
    """Calculates how similar the distributions of the various columns are."""

class DistanceDistribution(BaseMetric):
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...

class MutualInfo(BaseMetric):
    def __init__(self, formatter=...) -> None: ...
    @property
    def name(self) -> str: ...

class Autocorrelation(BaseMetric):
    def __init__(self, formatter=..., exclude_entity_col: bool = True) -> None: ...
    @property
    def name(self) -> str: ...
