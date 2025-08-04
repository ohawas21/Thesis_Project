from typing import List, Optional

from pandas import DataFrame as pdDataFrame
from sklearn.decomposition import FastICA as skICA

from ydata.quality.outlier.prototype import BaseProjectionOperator


class ICA(BaseProjectionOperator):
    """PCA projector."""

    def __init__(self):
        self.model = None
        self.columns = None
        self.n_components: int = 2

    def fit_transform(self, X: pdDataFrame, n_components: int = 2, columns: Optional[List[str]] = None) -> pdDataFrame:
        self = self.fit(X, n_components=n_components, columns=columns)
        X = self.transform(X)
        return X

    def fit(self, X: pdDataFrame, n_components: int = 2, columns: Optional[List[str]] = None):
        self.columns = columns
        self.n_components = n_components
        X_ = X[columns] if columns is not None else X
        for c in X_.columns:
            X_[c].fillna(X[c].mean(), inplace=True)
        self.model = skICA(
            n_components=self.n_components,
            random_state=42,
        )
        self.model.fit(X_)
        return self

    def transform(self, X: pdDataFrame) -> pdDataFrame:
        for c in X.columns:
            X[c].fillna(X[c].mean(), inplace=True)
        X_ = self.model.transform(X)
        X_ = pdDataFrame(X_, index=X.index)
        return X_

    def summary(self) -> dict:
        return {
            **super().summary(),
            **{
                "n_components": self.n_components
            }
        }
