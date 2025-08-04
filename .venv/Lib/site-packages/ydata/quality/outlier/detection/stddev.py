from pandas import DataFrame as pdDataFrame

from ydata.quality.outlier.prototype import BaseDetectionOperator


class StandardDeviation(BaseDetectionOperator):

    def __init__(self, sigma: float = 3, threshold: float = 0.5):
        self.sigma: float = sigma
        self.threshold: float = threshold

    def fit_predict(self, X: pdDataFrame) -> pdDataFrame:
        self = self.fit(X)
        y = self.predict(X)
        return y

    def fit(self, X: pdDataFrame):
        return self

    def predict(self, X: pdDataFrame) -> pdDataFrame:
        std = abs(X.std(axis=0))
        y = abs(X.mean(axis=0).T - X) / std
        y = y.fillna(0.)
        y = (y > self.sigma).sum(axis=1) / X.shape[1]
        return y

    def summary(self) -> dict:
        return {
            **super().summary(),
            **{
                "sigma": self.sigma,
                "threshold": self.threshold
            }
        }
