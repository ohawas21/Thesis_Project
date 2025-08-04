"""Method with utility preprocessing transformers that are shared between YData
modules: synthesizers, report and quality."""
from typing import Union

from pandas import DataFrame as pdDataframe
from sklearn.base import BaseEstimator, TransformerMixin


# @typechecked
class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base Transformer."""

    def __init__(self):
        super(BaseEstimator, self).__init__()

    def fit(self, X: pdDataframe, y=None):
        return self

    def transform(self, X: pdDataframe):
        """Returns the original series."""
        return X

    def inverse_transform(self, X: pdDataframe):
        """Returns the original series."""
        return X


class DropColumns(BaseTransformer):
    """Transformer class to drop columns that shouldn't considered into a
    pipeline."""

    def __init__(self):
        super(DropColumns, self).__init__()

    def _validate_data(self, X, y=None, columns=None, **check_params):
        check_cols = all(col in X.columns for col in self.columns)
        assert (
            check_cols
        ), "The provided columns do not match the input dataset. Please validate your inputs."
        self.columns = columns

    def fit(self, X: pdDataframe, columns: Union[str, list]):
        self._validate_data(X, columns=columns)
        return self

    def transform(self, X: pdDataframe) -> pdDataframe:
        result = X.copy()
        return result[result.columns[~result.columns.isin(self.columns)]]
