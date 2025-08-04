"""Preprocessors base classes."""
from sklearn.base import BaseEstimator, TransformerMixin

from ydata.__serializer import SerializerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin, SerializerMixin):
    def __init__(self, metadata):
        self._metadata = metadata
        self._pipeline = None

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def metadata(self):
        return self._metadata

    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError
