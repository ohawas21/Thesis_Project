from typing import List

import hdbscan
from pandas import DataFrame as pdDataFrame
from pandas import Index as pdIndex

from ydata.quality.outlier.prototype import (DEFAULT_CLUSTER_COL, DEFAULT_IS_OUTLIER_SCORE_COL, BaseClusteringOperator,
                                             OutlierCluster)


class HDBSCANClustering(BaseClusteringOperator):
    """Clustering using HDBSCAN."""

    def __init__(self, outlier_col=DEFAULT_IS_OUTLIER_SCORE_COL):
        self.outlier_col = outlier_col
        self.model = None

    def fit(self, X: pdDataFrame):
        # TODO: Allow the configuration externally
        min_samples = min(15, X.shape[0])
        self.model = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=2, prediction_data=True, allow_single_cluster=True)
        self.model = self.model.fit(X)
        return self

    def predict(self, X: pdDataFrame, cluster_col: str = DEFAULT_CLUSTER_COL) -> List[OutlierCluster]:
        y, _ = hdbscan.approximate_predict(self.model, X)
        y = pdDataFrame(y, columns=[cluster_col], index=X.index)
        y = y.groupby(cluster_col).groups
        clusters = list(map(lambda i: OutlierCluster(
            index=pdIndex(i)), y.values()))
        return clusters
