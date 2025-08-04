from _typeshed import Incomplete
from ydata.__models._cartmodel._methods import BaseMethod
from ydata.utils.random import RandomSeed as RandomSeed

class SampleMethod(BaseMethod):
    y_dtype: Incomplete
    smoothing: Incomplete
    proper: Incomplete
    random_state: Incomplete
    def __init__(self, y_dtype, smoothing: bool = False, proper: bool = False, random_state: RandomSeed = None, *args, **kwargs) -> None: ...
    values: Incomplete
    def fit(self, y: Incomplete | None = None, *args, **kwargs) -> None: ...
    def predict(self, X_test, random_state: RandomSeed = None, *args, **kwargs): ...
