"""
    Random utility method for synthesizers and Dataset
"""
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

from typing import TypeAlias

import dask.array as da
from numpy.random import RandomState, seed

RandomSeed: TypeAlias = int | RandomState | None


def set_random_state(random_state: int):
    seed(random_state)
    da.random.default_rng(random_state)
