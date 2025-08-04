from pandas import DataFrame as pdDataFrame
from pyod.models.iforest import IForest

from ydata.quality.outlier.prototype import BaseDetectionOperator


class IsolationForest(BaseDetectionOperator):

    def __init__(self):
        """
        References:
            1. Isolation forest
            Liu, F.T., Ting, K.M. and Zhou, Z.H.
            In International Conference on Data Mining, pp. 413-422. IEEE, 2008
        """
        self.model = None

    def fit_predict(self, X: pdDataFrame) -> pdDataFrame:
        self = self.fit(X)
        y = self.predict(X)
        return y

    def fit(self, X: pdDataFrame):
        self.model = IForest()
        self.model.fit(X)
        return self

    def predict(self, X: pdDataFrame) -> pdDataFrame:
        y = self.model.decision_function(X)
        return y
