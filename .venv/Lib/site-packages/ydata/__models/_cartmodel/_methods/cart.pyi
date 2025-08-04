from _typeshed import Incomplete
from dataclasses import dataclass
from ydata.__models._cartmodel._methods import BaseMethod
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed as RandomSeed

@dataclass
class CartHyperparameterConfig:
    min_samples_leaf: int = ...
    max_depth: int = ...
    splitter: str = ...
    def __init__(self, min_samples_leaf=..., max_depth=..., splitter=...) -> None: ...

class CartHyperparameterStrategy:
    def __call__(self, X, y, dtypes, y_dtype) -> CartHyperparameterConfig: ...

class CARTMethod(BaseMethod):
    y_dtype: Incomplete
    smoothing: Incomplete
    proper: Incomplete
    minibucket: Incomplete
    random_state: Incomplete
    cart: Incomplete
    hyperparam_strat: Incomplete
    def __init__(self, y_dtype, smoothing: bool = False, proper: bool = False, minibucket: int = 5, random_state: RandomSeed = None, *args, **kwargs) -> None:
        """
        proper: bool. For proper synthesis (proper=TRUE) a CART model is fitted to a bootstrapped sample of the original data
        smoothing: bool. To define whether smoothing should be applied to numerical variables
        """
    leaves_y_dict: Incomplete
    def fit(self, X, y, dtypes: dict = None): ...
    def predict(self, X, dtypes: dict = None, random_state: RandomSeed = None): ...

class SeqCARTMethod(BaseMethod):
    y_dtype: Incomplete
    smoothing: Incomplete
    proper: Incomplete
    minibucket: Incomplete
    random_state: Incomplete
    cluster: Incomplete
    order: Incomplete
    cart: Incomplete
    hyperparam_strat: Incomplete
    dummy: Incomplete
    def __init__(self, y_dtype: DataType, smoothing: bool = False, proper: bool = False, minibucket: int = 5, random_state: RandomSeed = None, *args, **kwargs) -> None:
        """
        proper: bool. For proper synthesis (proper=TRUE) a CART model is fitted to a bootstrapped sample of the original data
        smoothing: bool. To define whether smoothing should be applied to numerical variables
        """
    leaves_y_dict: Incomplete
    def fit(self, X, y, autoregressive: bool = False, dtypes: dict = None, *args, **kwargs): ...
    def predict(self, X, autoregressive: bool, dtypes: dict = None, cluster: list = None, random_state: RandomSeed = None): ...
