from pandas import DataFrame as pdDataFrame
from pyod.models.hbos import HBOS as odHBOS

from ydata.quality.outlier.prototype import BaseDetectionOperator


class HBOS(BaseDetectionOperator):

    def __init__(self):
        """
        References:
            1. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm.
            Goldstein, M. and Dengel, A.
            In KI-2012: Poster and Demo Track, pp.59-63, 2012
        """
        self.model = None

    def fit_predict(self, X: pdDataFrame) -> pdDataFrame:
        self = self.fit(X)
        y = self.predict(X)
        return y

    def fit(self, X: pdDataFrame):
        self.model = odHBOS()  # skIsolationForest(contamination=0.01)
        self.model.fit(X)
        return self

    def predict(self, X: pdDataFrame) -> pdDataFrame:
        y = self.model.predict_proba(X)[:, 0]
        # HBOS.predict_proba behaves strangely and seems to return the probably to not be an outlier.
        # The score is still in [0,1] so we need to transform it to [-1, 0]
        y = - (2 * y - 1)
        return y
