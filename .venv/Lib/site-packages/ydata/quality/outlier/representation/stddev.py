from typing import Any

import matplotlib.pyplot as plt
from pandas import DataFrame as pdDataFrame

from ydata.quality.outlier.prototype import BaseOperator


class StandardDeviationRepresentation(BaseOperator):

    def __init__(self, sigma: float = 3, threshold: float = 0.5):
        self.sigma: float = sigma
        self.threshold: float = threshold

    def fit_transform(self, X: pdDataFrame) -> pdDataFrame:
        self = self.fit(X)
        y = self.transform(X)
        return y

    def fit(self, X: pdDataFrame):
        return self

    def transform(self, X: pdDataFrame) -> pdDataFrame:
        std = abs(X.std(axis=0).T).T
        y = abs(X.mean(axis=0).T - X) / std
        y = y.fillna(0.)
        return y

    @staticmethod
    def _set_visibility(
        axis: Any, tick_mark: str = "none"
    ) -> Any:
        for anchor in ["top", "right", "bottom", "left"]:
            axis.spines[anchor].set_visible(False)
        axis.xaxis.set_ticks_position(tick_mark)
        axis.yaxis.set_ticks_position(tick_mark)
        return axis

    def plot(self, X: pdDataFrame, ax=None):
        # Filter column with no outliers
        X = X[X['outlier_score'] > self.threshold]
        X = X.dropna()  # [columns].dropna()
        X = X.drop(columns=['cluster'], errors='ignore')
        columns = [c for c in X.columns if X[X[c] > self.sigma]
                   [c].shape[0] > self.threshold]

        X = X[columns + ['outlier_score']]
        if ax is None:
            _, ax = plt.subplots(figsize=[12, 5])
        for i, c in enumerate(X):
            col = X.columns[i]
            if col in columns:
                x = X[
                    (X[col] > self.sigma)
                    & (X['outlier_score'] > self.threshold)
                ]
                ax.scatter(
                    y=[i] * x.shape[0],
                    x=x.iloc[:, i],
                    color='#166FF5',
                    alpha=0.2 if x.shape[0] > 25 else 0.5,
                    s=x.outlier_score.apply(lambda v: max(
                        10 if x.shape[0] > 25 else 30, 2000 * v**10)),
                )
        padding = 0.3
        ylim = ax.get_ylim()
        ax.set_ylim((ylim[0] - padding, ylim[1] + padding))

        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns)
        ax.axvline(x=self.sigma, color='red', linestyle='dotted')
        self._set_visibility(ax)

    def summary(self) -> dict:
        return {
            **super().summary(),
            **{
                "sigma": self.sigma,
                "threshold": self.threshold
            }
        }
