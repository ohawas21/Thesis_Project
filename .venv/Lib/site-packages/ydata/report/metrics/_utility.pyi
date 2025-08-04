from _typeshed import Incomplete
from ydata.metadata import Metadata as Metadata
from ydata.report.metrics import MetricType
from ydata.report.metrics.base_metric import BaseMetric

tstr_mapper: Incomplete
DECIMAL_PRECISION: int

class PredictiveScore(BaseMetric):
    def __init__(self, formatter=..., exclude_entity_col: bool = True) -> None: ...
    @property
    def name(self) -> str: ...

class TSTR(PredictiveScore):
    def __init__(self, formatter=...) -> None: ...

class TSTRTimeseries(PredictiveScore):
    def __init__(self, formatter=..., exclude_entity_col: bool = True) -> None: ...

def feature_importance(df_real, df_synth, target, visualize: bool = False):
    """
    target- temporary
    Extract information about the features of both dataset. To see if they have the same importance.
    Outputs:
        feat_scores: pdDataframe containing best features per dataset with relative score.
    """

class QScore(BaseMetric):
    """QScore returns The QScore measures the downstream performance of the
    synthetic data by running many random aggregation-based queries on both the
    synthetic and original datasets and then scoring the similarity of their
    returns.

    Args:
    max_cols (int): number of columns to group by.
    n_queries (int): number of random queries to run.
    n_bins (int): number of bins to discretize numerical columns. Defaults to 100.
    compute_penalty (bool): enables the penalty computation. Defaults to False.
    """
    max_cols: Incomplete
    n_queries: Incomplete
    n_bins: Incomplete
    max_categories: Incomplete
    compute_penalty: Incomplete
    def __init__(self, formatter=..., max_cols: int = 2, n_queries: int = 1000, n_bins: int = 100, compute_penalty: bool = False) -> None: ...
    @property
    def name(self) -> str: ...
    @staticmethod
    def penalty_score(matched_df, real_prematch, synth_prematch): ...

class FeatureImportance(BaseMetric):
    def __init__(self, formatter=..., include_plot: bool = True) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def type(self) -> MetricType: ...
    def feat_importance_score(self, importance_real, importance_synth, max_features: int = 10): ...
