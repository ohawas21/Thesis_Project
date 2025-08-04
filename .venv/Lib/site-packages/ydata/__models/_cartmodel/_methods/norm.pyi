from _typeshed import Incomplete
from ydata.__models._cartmodel._methods import BaseMethod
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed as RandomSeed

class NormMethod(BaseMethod):
    y_dtype: Incomplete
    smoothing: Incomplete
    proper: Incomplete
    random_state: Incomplete
    alpha: Incomplete
    norm: Incomplete
    def __init__(self, y_dtype: DataType, smoothing: bool = False, proper: bool = False, random_state: RandomSeed = None, ridge: float = 1e-05, *args, **kwargs) -> None:
        """
        proper: bool. For proper synthesis (proper=TRUE) a CART model is fitted to a bootstrapped sample of the original data
        smoothing: bool. To define whether smoothing should be applied to numerical variables
        """
    sigma: Incomplete
    def fit(self, X, y, dtypes: dict = None, *args, **kwargs): ...
    def predict(self, X_test, dtypes: dict = None, random_state: RandomSeed = None): ...
