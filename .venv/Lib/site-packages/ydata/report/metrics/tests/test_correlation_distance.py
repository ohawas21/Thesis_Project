import pytest
from pandas import DataFrame as pdDataFrame

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.report.metrics._metrics import DistanceCorrelation


def test_correlation_distance():
    original = pdDataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                            "b": [1.0, 2.0, 3.0, 4.0, 6.0]})
    synthetic = pdDataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                            "b": [1.0, 3.0, 8.0, 15.0, 6.0]})
    dataset = Dataset(original)
    metadata = Metadata(dataset)
    distance_corr = DistanceCorrelation()

    assert distance_corr.distance_correlation(
        original, synthetic, metadata) == pytest.approx(0.7)
