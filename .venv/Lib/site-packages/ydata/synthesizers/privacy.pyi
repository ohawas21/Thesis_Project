from _typeshed import Incomplete
from dataclasses import dataclass
from pandas import DataFrame as pdDataFrame
from ydata.datascience.common import PrivacyLevel
from ydata.utils.random import RandomSeed as RandomSeed

@dataclass
class PrivacyParameters:
    """Differential privacy parameters."""
    epsilon: float
    delta: float
    def __init__(self, epsilon, delta) -> None: ...

class DifferentialPrivacyLayer:
    """Differential privacy layer."""
    random_state: Incomplete
    def __init__(self, time_series: bool = False, random_state: RandomSeed = None) -> None:
        """Initialize the differential privacy layer.

        Args:
            time_series (bool): Whether the privacy layer will be applied to time series data.
        """
    def apply(self, X: pdDataFrame, privacy_level: PrivacyLevel, input_dtypes: dict):
        """Apply the differential privacy layer to a dataset.

        Args:
            X (pdDataFrame): Dataset that will receive the differential privacy noise.
            privacy_level (PrivacyLevel): Privacy level.
            input_dtypes: (dict): Data type of each column.
        """
