from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.dataset import Dataset as Dataset
from ydata.metadata import Metadata as Metadata
from ydata.report.reports.report_type import ReportType

class SyntheticDataMetrics:
    """Storage class for for Synthetic Data Fidelity, Utility and Privacy
    metrics.

    Args:
        report_type (ReportType): Whether to calculate the metrics for 'tabular' or 'timeseries' data. Defaults to 'tabular'.
        real: Dataset object with the real data to be evaluated
        synth: Dataset object with the synthetic data to be evaluated
    """
    creation_date: Incomplete
    target: Incomplete
    data_types: Incomplete
    metadata: Incomplete
    anonymized_cols: Incomplete
    safe_mode: Incomplete
    stats_summary: Incomplete
    non_anonymized_cols: Incomplete
    entity_cols: Incomplete
    def __init__(self, real: Dataset, synth: Dataset, data_types: dict, anonymized_cols: list[str], safe_mode: bool = True, training_data: Dataset | None = None, target: str | None = None, report_type: ReportType | str = ..., metadata: Metadata = None) -> None: ...
    def get_utility_metrics(self, real: pdDataFrame, synth: pdDataFrame, training_data: pdDataFrame | None = None):
        """Get utility metrics.

        - TSTR
        - TSTR for time-series
        - Discriminator
        """
    def get_fidelity_metrics(self, real: pdDataFrame, synth: pdDataFrame, training_data: pdDataFrame | None = None):
        """Univariate consistency metrics.

        - Statistical distribution and tests
        - Distribution plots for visual validation
        - Features autocorrelation (specific for time-series)
        Global consistency metrics
        - Correlation matrix distances
        - PCA and UMAP 2D plots for visual validation
        - Mutual information (specific for time-series)
        """
    def get_privacy_metrics(self, real: pdDataFrame, synth: pdDataFrame, training_data: pdDataFrame | None = None):
        """Calculate privacy metrics."""
    def get_info_metrics(self):
        """Calculates general info metrics."""
    def get_anonymized_metrics(self):
        """Obtains info about the anonymized columns."""
    def get_percentage_failed_metrics(self, privacy: dict, fidelity: dict, utility: dict): ...
    def get_percentage_failed(self, metrics: dict): ...
    def get_error_logs(self, metrics: dict): ...
    def evaluate(self):
        """Calculate metrics for synthetic data quality between provided real
        and synth datasets.

        Returns:
            calculated metrics (dict)
        """
