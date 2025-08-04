from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.quality.outlier.prototype import BaseDetectionOperator

class ECOD(BaseDetectionOperator):
    model: Incomplete
    def __init__(self) -> None:
        """
        References:
            1. ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.
            Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, H. G.
            IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.
        """
    def fit_predict(self, X: pdDataFrame) -> pdDataFrame: ...
    def fit(self, X: pdDataFrame): ...
    def predict(self, X: pdDataFrame) -> pdDataFrame: ...
