from _typeshed import Incomplete
from sklearn.base import BaseEstimator
from ydata.__serializer import SerializerMixin

class BaseSynthesizer(BaseEstimator, SerializerMixin):
    def fit(self, X, y: Incomplete | None = None, **kwargs) -> None: ...
    def sample(self, n_samples: int = 1): ...
