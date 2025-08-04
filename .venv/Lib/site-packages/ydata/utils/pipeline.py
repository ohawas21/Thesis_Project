"""Utilities for data processing pipelines."""
from sklearn.base import BaseEstimator, TransformerMixin


# Define custom transformer
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

    def inverse_transform(self, X):
        return X
