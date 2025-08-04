from _typeshed import Incomplete
from pandas import DataFrame as pdDataframe
from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base Transformer."""
    def __init__(self) -> None: ...
    def fit(self, X: pdDataframe, y: Incomplete | None = None): ...
    def transform(self, X: pdDataframe):
        """Returns the original series."""
    def inverse_transform(self, X: pdDataframe):
        """Returns the original series."""

class DropColumns(BaseTransformer):
    """Transformer class to drop columns that shouldn't considered into a
    pipeline."""
    def __init__(self) -> None: ...
    def fit(self, X: pdDataframe, columns: str | list): ...
    def transform(self, X: pdDataframe) -> pdDataframe: ...
