import random
from typing import List

import numpy as np
from numpy import max, min
from pandas import to_numeric

from ydata.__models._cartmodel._methods import BaseMethod
from ydata.__models._cartmodel._methods.utils import proper, smooth
from ydata.utils.data_types import DataType, VariableType
from ydata.utils.random import RandomSeed

# TODO modde self.perturb to a Listionary, one values for each variable


class PerturbMethod(BaseMethod):
    def __init__(
        self,
        y_dtype: DataType,
        smoothing: bool = False,
        proper: bool = False,
        random_state: RandomSeed = None,
        *args,
        **kwargs
    ):

        self.y_dtype = y_dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state

    def fit(
        self,
        X=None,
        y=None,
        dtypes: List = None,
        autoregressive: List = None,
        *args,
        **kwargs
    ):

        if self.proper:
            y = proper(y=y)

        if self.y_dtype["datatype"] == DataType.DATE or y.dtype in [
            VariableType.INT.value,
            VariableType.FLOAT.value,
        ]:
            if self.y_dtype["datatype"] == DataType.DATE:
                y = to_numeric(y)
            self.x_real_min, self.x_real_max = min(y), max(y)
            self.q = (np.min(y.diff().dropna()), np.max(y.diff().dropna()))
        self.values = y.copy()
        self.diff = y.diff().copy()

    def predict(self, X_test, autoregressive: List = None, random_state: RandomSeed = None, *args, **kwargs):
        y_pred = self.values.copy()
        if y_pred.dtype == VariableType.INT.value or y_pred.dtype in [
            VariableType.DATE.value,
            VariableType.DATETIME.value,
        ]:
            delta = int(random.uniform(self.q[0], self.q[1]))
        else:
            delta = random.uniform(self.q[0], self.q[1])

        y_pred[1] += delta
        y_pred[2:] += self.diff[2:]

        if self.y_dtype["datatype"] == DataType.DATE:
            y_pred = y_pred.sort_values(ascending=True).reset_index(drop=True)

        if self.smoothing and self.y_dtype["datatype"] in [
            DataType.NUMERICAL,
            DataType.DATE,
        ]:
            y_pred = smooth(
                self.y_dtype["datatype"], y_pred, self.x_real_min, self.x_real_max
            )
        return y_pred.values
