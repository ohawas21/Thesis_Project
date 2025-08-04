from _typeshed import Incomplete
from ydata.__models._cartmodel._methods import NormMethod

class NormRankMethod(NormMethod):
    """Adapted from norm by carrying out regression on Z scores from ranks
    predicting new Z scores and then transforming back."""
    sigma: Incomplete
    y_sorted: Incomplete
    def fit(self, X, y, dtypes: dict = None): ...
    def predict(self, X_test, dtypes: dict = None): ...
