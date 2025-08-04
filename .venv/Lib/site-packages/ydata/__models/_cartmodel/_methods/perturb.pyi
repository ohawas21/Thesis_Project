from _typeshed import Incomplete
from ydata.__models._cartmodel._methods import BaseMethod
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed as RandomSeed

class PerturbMethod(BaseMethod):
    y_dtype: Incomplete
    smoothing: Incomplete
    proper: Incomplete
    random_state: Incomplete
    def __init__(self, y_dtype: DataType, smoothing: bool = False, proper: bool = False, random_state: RandomSeed = None, *args, **kwargs) -> None: ...
    q: Incomplete
    values: Incomplete
    diff: Incomplete
    def fit(self, X: Incomplete | None = None, y: Incomplete | None = None, dtypes: list = None, autoregressive: list = None, *args, **kwargs): ...
    def predict(self, X_test, autoregressive: list = None, random_state: RandomSeed = None, *args, **kwargs): ...
