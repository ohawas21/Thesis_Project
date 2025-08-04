"""File to define the sample synthesis method."""
from numpy import max, min
from numpy.random import default_rng

from ydata.__models._cartmodel._methods import BaseMethod
from ydata.__models._cartmodel._methods.utils import proper, smooth
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed


class SampleMethod(BaseMethod):
    def __init__(
        self, y_dtype, smoothing=False, proper=False, random_state: RandomSeed = None, *args, **kwargs
    ):
        self.y_dtype = y_dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state

    def fit(self, y=None, *args, **kwargs):
        if self.proper:
            y = proper(y=y)
        if self.y_dtype == DataType.NUMERICAL:
            self.x_real_min, self.x_real_max = min(y), max(y)

        self.values = y.to_numpy()

    def predict(self, X_test, random_state: RandomSeed = None, *args, **kwargs):
        rng = default_rng(seed=random_state)
        n = X_test.shape[0]
        y_pred = rng.choice(self.values, size=n, replace=True)

        if self.smoothing and self.y_dtype == DataType.NUMERICAL:
            y_pred = smooth(self.y_dtype, y_pred,
                            self.x_real_min, self.x_real_max)

        return y_pred
