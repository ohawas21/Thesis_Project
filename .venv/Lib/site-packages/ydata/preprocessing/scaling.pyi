from _typeshed import Incomplete
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalFactors(BaseEstimator, TransformerMixin):
    """Transform numerical data based on factors.

    Factors are multiplicative scalars fitted/applied per numerical
    feature per each groupby value.
    """
    groupby: Incomplete
    default: Incomplete
    def __init__(self, groupby: str, factors: dict | None = None, default: bool = True) -> None:
        """
        Args:
            groupby (str): column in data used to map the factors
            factors (dict, optional): pre-trained factors to apply.
            default (bool, optional): convert factors to default dict of 1x factor on never-observed groupby values.
        """
    def fit(self, X, y: Incomplete | None = None): ...
    def transform(self, X, y: Incomplete | None = None):
        """Removes the factors influences."""
    def inverse_transform(self, X, y: Incomplete | None = None):
        """Reinstates the factors influences."""
    def save(self, file_path: str): ...
    @staticmethod
    def load(file_path: str): ...
