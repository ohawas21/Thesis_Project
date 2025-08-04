from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, unique
from math import ceil
from typing import Hashable

from ydata.dataset.dataset import Dataset
from ydata.metadata import Metadata
from ydata.metadata.column import Column

DEFAULT_SLICE_SIZE = 100
MIN_SLICE_SIZE = 10


@dataclass
class SliceInfo:
    mask: list
    name: Hashable
    size: int
    ratio: float
    info: dict | None = None


class SlicingStrategy(Iterable):
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None):
        self.metadata = metadata
        self.dtypes = dtypes if dtypes is not None else dict(metadata.columns)
        self.slices = []
        self._n_slices = None

    def __init_n_slices(self) -> None:
        if self._n_slices is None:
            self._n_slices = len(self.slices)

    def __iter__(self):
        self.__init_n_slices()
        return self.slices.__iter__()

    @property
    def n_slices(self) -> int:
        self.__init_n_slices()
        return self._n_slices


class NoSlicing(SlicingStrategy):
    def __init__(self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column] | None = None):
        super().__init__(X, metadata, dtypes)
        self.slices = self.__calculate_slices(X)

    def __calculate_slices(self, X: Dataset):
        N = X.shape[1]
        return [SliceInfo(X.columns, name=1, size=N, ratio=1.0)]


class SimpleSlicing(SlicingStrategy):
    def __init__(
        self,
        X: Dataset,
        metadata: Metadata,
        dtypes: dict[str, Column] = None,
        slice_size: int | float = DEFAULT_SLICE_SIZE,
    ):
        super().__init__(X, metadata, dtypes)
        self.slices = self.__calculate_slices(X, metadata, slice_size)

    def __calculate_slices(
        self, X: Dataset, metadata: Metadata, slice_size: int | float
    ):
        N = X.shape[1]
        if isinstance(slice_size, float):
            slice_size = int(ceil(slice_size * N))

        n_segments = (N // slice_size) + 1 if N > slice_size != 0 else 1

        if metadata.dataset_attrs is None:
            has_sortbykey = False
        else:
            try:
                has_sortbykey = len(metadata.dataset_attrs.sortbykey) > 0
            except AttributeError:
                has_sortbykey = False

        if has_sortbykey:
            sbk_index = [c for c in list(self.dtypes.keys())].index(
                metadata.dataset_attrs.sortbykey[0]
            )
        if n_segments in [0, 1]:
            slices = [SliceInfo(list(self.dtypes.keys()),
                                name=1, size=N, ratio=1.0)]
        else:
            if metadata.dataset_attrs is None:
                has_sortbykey = False
            else:
                try:
                    has_sortbykey = len(metadata.dataset_attrs.sortbykey) > 0
                except AttributeError:
                    has_sortbykey = False
            r = N % slice_size
            mask = range(N)
            slices = [range(e, e + (r if i == n_segments - 1 else slice_size))
                      for i, e in enumerate(mask[:N:slice_size])
                      ]
            slices = [
                SliceInfo(
                    [X.columns[e] for e in r],
                    name=i,
                    size=len(r),
                    ratio=float(len(r)) / N,
                )
                for i, r in enumerate(slices)
            ]
            if has_sortbykey:
                slices = [
                    s
                    if X.columns[sbk_index] in s.mask
                    else SliceInfo(
                        [X.columns[sbk_index]] + s.mask,
                        name=s.name,
                        size=s.size + 1,
                        ratio=(s.size + 1) / N,
                    )
                    for i, s in enumerate(slices)
                ]
        return slices


@unique
class Strategy(Enum):
    NONE = NoSlicing
    SIMPLE = SimpleSlicing
