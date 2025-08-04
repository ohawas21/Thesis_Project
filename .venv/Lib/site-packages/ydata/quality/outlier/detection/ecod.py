from pandas import DataFrame as pdDataFrame
from pyod.models.ecod import ECOD as odECOD

from ydata.quality.outlier.prototype import BaseDetectionOperator


class ECOD(BaseDetectionOperator):

    def __init__(self):
        """
        References:
            1. ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.
            Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, H. G.
            IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.
        """
        self.model = None

    def fit_predict(self, X: pdDataFrame) -> pdDataFrame:
        self = self.fit(X)
        y = self.predict(X)
        return y

    def fit(self, X: pdDataFrame):
        self.model = odECOD()
        self.model.fit(X)
        return self

    def predict(self, X: pdDataFrame) -> pdDataFrame:
        y = self.model.predict_proba(X)[:, 0]
        # ECOD.predict_proba behaves strangely and seems to return the probably to not be an outlier.
        # The score is still in [0,1] so we need to transform it to [-1, 0]
        y = - (2 * y - 1)
        return y
