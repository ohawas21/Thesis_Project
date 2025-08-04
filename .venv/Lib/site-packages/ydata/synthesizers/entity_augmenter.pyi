from _typeshed import Incomplete
from dataclasses import dataclass
from pandas import DataFrame as pdDataFrame
from typing import Literal
from ydata.metadata import Metadata as Metadata
from ydata.utils.random import RandomSeed as RandomSeed

@dataclass
class SmoothingConfig:
    """Smoothing Configuration.

    enabled (bool): enable or not the smoothing. window (float): time
    window to use for smoothing. Expressed as a fraction of the original
    dataset size. degree (int): degree of the polynomial to use for
    smoothing. derivative (int): order of the derivative to consider for
    smoothing
    """
    enabled: bool = ...
    window: float = ...
    degree: int = ...
    derivative: int = ...
    def __init__(self, enabled=..., window=..., degree=..., derivative=...) -> None: ...

@dataclass
class FidelityConfig:
    """Fidelity Configuration.

    strategy (str): strategy to use for the behavioral noise. noise
    (float): noise amplitude expressed as a fraction of the standard
    deviation. Can theoreically be higher than 1.
    """
    strategy: Literal['gaussian', 'uniform'] = ...
    noise: float = ...
    def __init__(self, strategy=..., noise=...) -> None: ...

class EntityAugmenter:
    nrows: Incomplete
    n_entities: Incomplete
    original_sbk: Incomplete
    pert_blocks: Incomplete
    gauss_models: Incomplete
    pivot_columns: Incomplete
    all_columns: Incomplete
    sortbykey: Incomplete
    entities: Incomplete
    smoothing_strategy: Incomplete
    smoother_per_column: bool
    fidelity_strategy: Incomplete
    fidelity_per_column: bool
    gauss_config: Incomplete
    def __init__(self, X: pdDataFrame, metadata: Metadata, n_entities: int, pivot_columns: list[str]) -> None: ...
    def fit_block_bootstraper(self, block_name: str, X: pdDataFrame):
        """The entity augmenter works by bootstrapping the synthesizer with few
        columns called `pivot` obtained in few steps:

        1. Gaussianization of the original columns order in by sortbykey + entities - by this order we preserve all dependencies
        2. Sorting by entities + sortbykey
        3. Calculating the distribution of difference per entity
        4. Sort back to the exact same order to avoid issues with previous preprocessing steps done outside the entity augmenter
        """
