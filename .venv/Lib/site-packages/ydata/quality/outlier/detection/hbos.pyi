from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.quality.outlier.prototype import BaseDetectionOperator

class HBOS(BaseDetectionOperator):
    model: Incomplete
    def __init__(self) -> None:
        """
        References:
            1. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm.
            Goldstein, M. and Dengel, A.
            In KI-2012: Poster and Demo Track, pp.59-63, 2012
        """
    def fit_predict(self, X: pdDataFrame) -> pdDataFrame: ...
    def fit(self, X: pdDataFrame): ...
    def predict(self, X: pdDataFrame) -> pdDataFrame: ...
