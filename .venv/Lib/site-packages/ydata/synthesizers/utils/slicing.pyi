from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Hashable
from ydata.dataset.dataset import Dataset as Dataset
from ydata.metadata import Metadata as Metadata
from ydata.metadata.column import Column as Column

DEFAULT_SLICE_SIZE: int
MIN_SLICE_SIZE: int

@dataclass
class SliceInfo:
    mask: list
    name: Hashable
    size: int
    ratio: float
    info: dict | None = ...
    def __init__(self, mask, name, size, ratio, info=...) -> None: ...

class SlicingStrategy(Iterable):
    metadata: Incomplete
    dtypes: Incomplete
    slices: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None) -> None: ...
    def __iter__(self): ...
    @property
    def n_slices(self) -> int: ...

class NoSlicing(SlicingStrategy):
    slices: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None) -> None: ...

class SimpleSlicing(SlicingStrategy):
    slices: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] = None, slice_size: int | float = ...) -> None: ...

class Strategy(Enum):
    NONE = NoSlicing
    SIMPLE = SimpleSlicing
