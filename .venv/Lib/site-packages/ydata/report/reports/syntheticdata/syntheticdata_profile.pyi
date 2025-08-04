from _typeshed import Incomplete
from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset as Dataset
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.report.reports.report_type import ReportType

class SyntheticDataProfile:
    """Class to generate a Synthetic Data Profile."""
    MIN_ROWS: int
    LOW_ROWS: int
    METRIC_ERROR_VALUE: Incomplete
    SCORE_TO_LABEL: Incomplete
    report_type: Incomplete
    safe_mode: Incomplete
    anonymized_cols: Incomplete
    creation_date: Incomplete
    css: Incomplete
    def __init__(self, real: Dataset, synth: Dataset, metadata: Metadata, privacy_level: PrivacyLevel | int = ..., report_type: ReportType | str = ..., safe_mode: bool = True, data_types: dict | None = None, training_data: Dataset | None = None, target: str | Column | None = None, anonymized_cols: list[str] | None = None, synth_name: str | None = None) -> None:
        """Initialization of the SyntheticDataProfile class.

        Args:
            real (Dataset): original dataset containing an holdout not used to train the synthesizer.
            synth (Dataset): synthetically generated data samples.
            metadata (Metadata): metadata of the original dataset.
            privacy_level (PrivacyLevel): the level of privacy applied to the synthetic data. Defaults to 'HIGH_FIDELITY'.
            report_type (ReportType): whether to calculate a report for 'tabular' or 'timeseries' data. Defaults to 'tabular'.
            safe_mode (bool): whether to run in safe mode or not. If True, any exception in the metrics is handled gracefully
                without crashing the report. Defaults to True.
            data_types (dict, optional): propagates the specified data_types for the calculation of the quality metrics.
            training_data (Dataset, optional): original dataset used to train the synthesizer.
                If provided, used to calculate some metric scores (e.g. membership score)
            target (str, optional): if provided, propagates the specified target for the calculation of the quality metrics.
            anonymized_cols (list[str], optional): list of columns that are anonymized.
            synth_name (str, optional): if provided, add the synthesizer name to the report.
        """
    @property
    def report_status(self): ...
    def get_all_metrics(self):
        """Returns all types of metrics."""
    def get_summary(self):
        """Returns the summary metrics."""
    def get_fidelity_metrics(self):
        """Returns the fidelity metrics."""
    def get_utility_metrics(self):
        """Returns the utility metrics."""
    def get_privacy_metrics(self):
        """Returns the privacy metrics."""
    def generate_report(self, output_path: str = './ydata-report.pdf') -> None:
        """Generates a .pdf report with synthetic data quality metrics.

        Args:
            output_path (str, optional): output path for the .pdf file.
        """
    def display_notebook(self) -> None: ...
