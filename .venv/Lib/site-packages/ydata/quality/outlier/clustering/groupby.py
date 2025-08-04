from typing import List

from pandas import DataFrame as pdDataFrame
from pandas import Index as pdIndex

from ydata.quality.outlier.prototype import BaseClusteringOperator, OutlierCluster


class GroupByClustering(BaseClusteringOperator):
    """Cluster the outliers based on their score value."""

    def predict(self, X: pdDataFrame) -> List[OutlierCluster]:
        y = X[[self.outlier_col]]
        y = y.groupby(self.outlier_col).groups
        clusters = list(map(lambda i: OutlierCluster(
            index=pdIndex(i)), y.values()))
        return clusters
