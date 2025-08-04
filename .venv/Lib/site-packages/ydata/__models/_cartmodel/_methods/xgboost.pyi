from _typeshed import Incomplete
from ydata.__models._cartmodel._methods import BaseMethod
from ydata.utils.data_types import DataType

class XGBMethod(BaseMethod):
    dtype: Incomplete
    smoothing: Incomplete
    proper: Incomplete
    minibucket: Incomplete
    random_state: Incomplete
    length: Incomplete
    cluster: Incomplete
    one_hot_cat_cols: Incomplete
    order: Incomplete
    xgboost: Incomplete
    def __init__(self, dtype: DataType, smoothing: bool = False, proper: bool = False, minibucket: int = 5, random_state: int = None, *args, **kwargs) -> None: ...
    leaves_y_dict: Incomplete
    def fit(self, X, y, autoregressive: dict, *args, **kwargs): ...
    def predict(self, X_test, autoregressive: dict, *args, **kwargs): ...
