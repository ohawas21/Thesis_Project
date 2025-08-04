import functools
from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from multimethod import multimethod
from scipy.stats import chisquare

from ydata_profiling.config import Settings

@multimethod
def describe_categorical_1d(
    config: Settings, series: pd.Series, summary: dict
) -> Tuple[Settings, pd.Series, dict]:

    raise NotImplementedError()
