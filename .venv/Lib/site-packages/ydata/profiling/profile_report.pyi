from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from pathlib import Path
from visions import VisionsTypeset as VisionsTypeset
from ydata.dataset import Dataset
from ydata.profiling.model import BaseDescription
from ydata_profiling import ProfileReport as _ProfileReport
from ydata_profiling.config import Settings
from ydata_profiling.model.summarizer import BaseSummarizer as BaseSummarizer
from ydata_profiling.report.presentation.core import Root

metrics_logger: Incomplete

class ProfileReport(_ProfileReport):
    def __init__(self, dataset: Dataset | pdDataFrame | None = None, minimal: bool = False, explorative: bool = False, sensitive: bool = False, tsmode: bool = False, sortby: str | None = None, sample: dict | None = None, config_file: Path | str | None = None, lazy: bool = True, typeset: VisionsTypeset | None = None, summarizer: BaseSummarizer | None = None, config: Settings | None = None, outlier: bool = False, **kwargs) -> None: ...
    @property
    def summarizer(self) -> BaseSummarizer: ...
    @property
    def typeset(self) -> VisionsTypeset | None: ...
    @property
    def description_set(self) -> BaseDescription: ...
    @property
    def report(self) -> Root: ...
    def to_file(self, output_file: str | Path, silent: bool = True, html_breakdown: bool = False) -> None:
        """Write the report to a file.

        By default a name is generated.

        Args:
            output_file: The name or the path of the file to generate including the extension (.html, .json).
            silent: if False, opens the file in the default browser or download it in a Google Colab environment
        """
    def compare(self, other: ProfileReport, config: Settings | None = None) -> ProfileReport:
        """Compare this report with another ProfileReport
        Alias for:
        ```
        ydata_profiling.compare([report1, report2], config=config)
        ```
        See `ydata_profiling.compare` for details.

        Args:
            other: the ProfileReport to compare to
            config: the settings object for the merged ProfileReport. If `None`, uses the caller's config

        Returns:
            Comparison ProfileReport
        """
    def dumps(self, include_schema: bool = False) -> bytes:
        """Serialize ProfileReport and return bytes for reproducing
        ProfileReport or Caching.

        Args:
            include_schema (bool): True if the dataframe schema should be saved
        Returns:
            Bytes which contains hash of DataFrame, config, _description_set and _report
        """
    @staticmethod
    def loads(data: bytes) -> ProfileReport:
        """Deserialize the serialized report.

        Args:
            data: The bytes of a serialize ProfileReport object.
        Raises:
            ValueError: if ignore_config is set to False and the configs do not match.
        Returns:
            self
        """
    def save(self, output_file: Path | str, include_schema: bool = False):
        """Save the report to a file.

        Args:
            output_file: Path where to save the ProfileReport
            include_schema (bool): True if the dataframe schema should be saved
        """
    @staticmethod
    def load(path: Path, dataset: pdDataFrame | None = None) -> ProfileReport:
        """Load a ProfileReport from a file.

        Args:
            path: Path where to load the ProfileReport
            dataset (Optional[pd.DataFrame]): dataset to re-assign to the ProfileReport
        """
