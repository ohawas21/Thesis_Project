from typing import List

from pandas import DataFrame as pdDataFrame
from pandas import Index as pdIndex

from ydata.quality.outlier.prototype import BaseClusteringOperator, OutlierCluster


class IndividualClustering(BaseClusteringOperator):
    """Naive clustering which return a cluster per outlier elements."""

    def predict(self, X: pdDataFrame) -> List[OutlierCluster]:
        y = X.index
        clusters = list(map(lambda i: OutlierCluster(
            index=pdIndex([i])), y.values))
        return clusters
