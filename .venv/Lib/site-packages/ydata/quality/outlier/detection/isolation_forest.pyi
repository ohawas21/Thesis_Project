from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.quality.outlier.prototype import BaseDetectionOperator

class IsolationForest(BaseDetectionOperator):
    model: Incomplete
    def __init__(self) -> None:
        """
        References:
            1. Isolation forest
            Liu, F.T., Ting, K.M. and Zhou, Z.H.
            In International Conference on Data Mining, pp. 413-422. IEEE, 2008
        """
    def fit_predict(self, X: pdDataFrame) -> pdDataFrame: ...
    def fit(self, X: pdDataFrame): ...
    def predict(self, X: pdDataFrame) -> pdDataFrame: ...
