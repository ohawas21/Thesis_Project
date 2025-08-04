from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass
from pandas import Series as pdSeries
from typing import Hashable
from ydata.dataset.dataset import Dataset as Dataset
from ydata.metadata import Metadata as Metadata
from ydata.metadata.column import Column as Column
from ydata.utils.enum_utils import EnumToLiteralMixIn

DEFAULT_SEGMENT_SIZE: int
MIN_SEGMENT_SIZE: int

@dataclass
class SegmentInfo:
    mask: pdSeries
    name: Hashable
    size: int
    ratio: float
    info: dict | None = ...
    def __init__(self, mask, name, size, ratio, info=...) -> None: ...

class SegmentationStrategy(Iterable):
    metadata: Incomplete
    dtypes: Incomplete
    segments: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None) -> None: ...
    def __iter__(self): ...
    @property
    def n_segments(self) -> int: ...

class NoSegmentation(SegmentationStrategy):
    segments: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None) -> None: ...

class SimpleSegmentation(SegmentationStrategy):
    segments: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None, segment_size: int | float = ...) -> None: ...

class EntitySegmentation(SegmentationStrategy):
    segments: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, entity_col: str, dtypes: dict[str, Column] | None = None) -> None: ...

class NNSegmentation(SegmentationStrategy):
    segments: Incomplete
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None, columns: list[str] | None = None, min_cluster_size: int = ..., max_n_clusters: int = 10) -> None: ...

def iterative_nn(X, preprocessed_X, min_cluster_size: int = 10000, max_n_clusters: int = 10): ...

class Strategy(EnumToLiteralMixIn):
    NONE = NoSegmentation
    SIMPLE = SimpleSegmentation
    ENTITY = EntitySegmentation
    NN = NNSegmentation
