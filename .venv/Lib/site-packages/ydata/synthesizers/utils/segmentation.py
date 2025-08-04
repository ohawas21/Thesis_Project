import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from enum import unique
from math import ceil
from typing import Dict, Hashable, List, Optional, Union

from numpy import argpartition, concatenate, mean
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
from pandas import qcut
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MiniBatchKMeans

from ydata.dataset.dataset import Dataset
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.preprocessors.regular.preprocessor import DataProcessor
from ydata.synthesizers.exceptions import SegmentSizeWarning
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType
from ydata.utils.enum_utils import EnumToLiteralMixIn

DEFAULT_SEGMENT_SIZE = 150_000
MIN_SEGMENT_SIZE = 30_000

@dataclass
class SegmentInfo:
    mask: pdSeries
    name: Hashable
    size: int
    ratio: float
    info: Optional[Dict] = None


class SegmentationStrategy(Iterable):
    def __init__(
        self, X: Dataset, metadata: Metadata, dtypes: Optional[Dict[str, Column]] = None
    ):
        self.metadata = metadata
        self.dtypes = dtypes if dtypes is not None else dict(metadata.columns)
        self.segments = []
        self._n_segments = None

    def __init_n_segment(self) -> None:
        if self._n_segments is None:
            self._n_segments = len(self.segments)

    @staticmethod
    def _check_segment_size(segment_size) -> int:
        if segment_size < MIN_SEGMENT_SIZE:
            warnings.warn(
                f"The segment size is too small. Use the segment size of {MIN_SEGMENT_SIZE}",
                SegmentSizeWarning,
            )
        return max(MIN_SEGMENT_SIZE, segment_size)

    def __iter__(self):
        self.__init_n_segment()
        return self.segments.__iter__()

    @property
    def n_segments(self) -> int:
        self.__init_n_segment()
        return self._n_segments


class NoSegmentation(SegmentationStrategy):
    def __init__(
        self, X: Dataset, metadata: Metadata, dtypes: Optional[Dict[str, Column]] = None
    ):
        super().__init__(X, metadata, dtypes)
        self.segments = self.__calculate_segments()

    def __calculate_segments(self):
        N = self.metadata.summary["nrows"]
        return [SegmentInfo(pdSeries([True] * N), name=1, size=N, ratio=1.0)]


class SimpleSegmentation(SegmentationStrategy):
    def __init__(
        self,
        X: Dataset,
        metadata: Metadata,
        dtypes: Optional[Dict[str, Column]] = None,
        segment_size: Union[int, float] = DEFAULT_SEGMENT_SIZE,
    ):
        super().__init__(X, metadata, dtypes)
        self.segments = self.__calculate_segments(X, dtypes, segment_size)

    def __calculate_segments(
        self,
        X: Dataset,
        dtypes: Optional[Dict[str, Column]],
        segment_size: Union[int, float],
    ):
        N = self.metadata.summary["nrows"]
        S = pdDataFrame({"index": range(N)})

        if isinstance(segment_size, float):
            segment_size = int(ceil(segment_size * N))

        segment_size = self._check_segment_size(segment_size)
        n_segments = N // segment_size + (1 if N % segment_size else 0)
        if n_segments in [0, 1]:
            segments = [SegmentInfo(
                pdSeries([True] * N), name=1, size=N, ratio=1.0)]
        else:
            segments = []
            r_segments = qcut(S["index"], n_segments)
            for s in r_segments.cat.categories:
                mask = r_segments == s
                size = mask.sum()
                segments.append(
                    SegmentInfo(mask, name=s, size=size, ratio=float(size) / N)
                )
        return segments


class EntitySegmentation(SegmentationStrategy):
    def __init__(
        self,
        X: Dataset,
        metadata: Metadata,
        entity_col: str,
        dtypes: Optional[Dict[str, Column]] = None,
    ):
        super().__init__(X, metadata, dtypes)
        self.segments = self.__calculate_segments(X, entity_col=entity_col)

    def __calculate_segments(self, X: Dataset, entity_col: str):
        r_segments = X[entity_col]
        segments = []
        for s in r_segments.unique():
            mask = r_segments == s
            size = mask.sum()
            segments.append(
                SegmentInfo(mask, name=s, size=size,
                            ratio=float(size) / X.shape[0])
            )
        return segments


class NNSegmentation(SegmentationStrategy):
    def __init__(
        self,
        X: Dataset,
        metadata: Metadata,
        dtypes: Optional[Dict[str, Column]] = None,
        columns: Optional[List[str]] = None,
        min_cluster_size: int = MIN_SEGMENT_SIZE,
        max_n_clusters: int = 10,
    ):
        super().__init__(X, metadata, dtypes)
        self.segments = self.__calculate_segments(
            X, columns, min_cluster_size, max_n_clusters
        )

    def __calculate_segments(
        self,
        X: Dataset,
        columns: Optional[List[str]],
        min_cluster_size: int,
        max_n_clusters: int,
    ):
        if columns is None:
            columns = self.dtypes.keys()
        data_types = {k: v.datatype for k,
                      v in self.dtypes.items() if k in columns}
        preprocessor = DataProcessor(
            num_cols=[k for k, v in data_types.items() if v ==
                      DataType.NUMERICAL],
            cat_cols=[k for k, v in data_types.items() if v in
                      CATEGORICAL_DTYPES],
            dt_cols=[k for k, v in data_types.items() if v == DataType.DATE],
        )
        preprocessed_X = preprocessor.fit_transform(X)
        r_segments = iterative_nn(
            X,
            preprocessed_X,
            min_cluster_size=min_cluster_size,
            max_n_clusters=max_n_clusters,
        )
        segments = []
        for s in r_segments.unique():
            mask = r_segments == s
            size = mask.sum()
            segments.append(
                SegmentInfo(mask, name=s, size=size,
                            ratio=float(size) / X.shape[0])
            )
        return segments


def iterative_nn(
    X, preprocessed_X, min_cluster_size: int = 10000, max_n_clusters: int = 10
):
    if len(preprocessed_X) < min_cluster_size:
        return pdSeries([1] * len(preprocessed_X), index=X.index)

    kmeans = MiniBatchKMeans(n_clusters=max_n_clusters).fit(preprocessed_X)
    segments = pdSeries(kmeans.predict(preprocessed_X), index=X.index)

    centroids = kmeans.cluster_centers_
    cluster_names = list(range(centroids.shape[0]))
    cluster_sizes = [
        len(segments[segments == cluster_name]) for cluster_name in cluster_names
    ]
    small_cluster_indexes = [
        i for i, v in enumerate(cluster_sizes) if v < min_cluster_size
    ]

    while len(small_cluster_indexes) != 0:
        centroid_distances = squareform(pdist(centroids, "euclidean"))
        small_cluster_index = small_cluster_indexes.pop(0)
        small_cluster_name = cluster_names[small_cluster_index]
        small_cluster_distances = centroid_distances[small_cluster_index]
        nearest_neigh = cluster_names[argpartition(
            small_cluster_distances, 1)[1]]
        segments[segments == small_cluster_name] = nearest_neigh

        cluster_names.remove(small_cluster_name)
        cluster_sizes = [
            len(segments[segments == cluster_name]) for cluster_name in cluster_names
        ]
        small_cluster_indexes = [
            i for i, v in enumerate(cluster_sizes) if v < min_cluster_size
        ]
        centroids = concatenate(
            [
                mean(preprocessed_X[segments ==
                     cluster_name], axis=0).reshape(1, -1)
                for cluster_name in cluster_names
            ],
            axis=0,
        )

    return segments


@unique
class Strategy(EnumToLiteralMixIn):
    NONE = NoSegmentation
    SIMPLE = SimpleSegmentation
    ENTITY = EntitySegmentation
    NN = NNSegmentation
