"""File to define the empy method logic."""
from typing import List

from numpy import empty, nan

from ydata.__models._cartmodel._methods import BaseMethod


class EmptyMethod(BaseMethod):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        n = X.shape[0]
        y_pred = empty(n) * nan
        return y_pred


class SeqEmptyMethod(BaseMethod):
    def __init__(self, *args, **kwargs):
        self.values = None

    def fit(self, X, y, autoregressive: List = None, *args, **kwargs):
        self.values = y.to_numpy()

    def predict(self, X, autoregressive: List = None, *args, **kwargs):
        return self.values
