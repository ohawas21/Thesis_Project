from sklearn.base import BaseEstimator

from ydata.__serializer import SerializerMixin


class BaseSynthesizer(BaseEstimator, SerializerMixin):
    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def sample(self, n_samples: int = 1):
        raise NotImplementedError
