from _typeshed import Incomplete
from numpy import array as array
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from ydata.__serializer import SerializerMixin

class TimeSeriesSynthesizerPreprocessor(BaseEstimator, TransformerMixin, SerializerMixin):
    num_cols: Incomplete
    dt_cols: Incomplete
    num_pipeline: Incomplete
    def __init__(self, *, num_cols: list[str] = None, cat_cols: list[str] = None, dt_cols: list[str] = None) -> None: ...
    def fit(self, X: DataFrame | array): ...
    def transform(self, X: DataFrame | array): ...
    def inverse_transform(self, X): ...

class IndexBasedTSSynthesizerPreprocessor(TimeSeriesSynthesizerPreprocessor):
    def fit_transform(self, X: DataFrame, index: str = None): ...
