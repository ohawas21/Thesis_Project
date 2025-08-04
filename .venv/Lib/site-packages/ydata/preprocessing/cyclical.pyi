from _typeshed import Incomplete
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicalFeatures(BaseEstimator, TransformerMixin):
    """Apply sin+cos transformation for features with a natural cyclical nature
    (e.g. hour of day)."""
    maxs: Incomplete
    drop: Incomplete
    cols: Incomplete
    def __init__(self, maxs: list[float] | None = None, drop: bool = True) -> None:
        """Transformer to convert numerical cyclical features (sin + cos).

        Args:
            maxs (List[float], optional): list of maximum feature values of columns.
            drop (bool): drop original columns after transformation
        """
    @staticmethod
    def sin(x, max): ...
    @staticmethod
    def cos(x, max): ...
    def fit(self, X, y: Incomplete | None = None): ...
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X, y: Incomplete | None = None): ...
