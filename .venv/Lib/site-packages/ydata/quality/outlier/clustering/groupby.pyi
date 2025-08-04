from pandas import DataFrame as pdDataFrame
from ydata.quality.outlier.prototype import BaseClusteringOperator, OutlierCluster

class GroupByClustering(BaseClusteringOperator):
    """Cluster the outliers based on their score value."""
    def predict(self, X: pdDataFrame) -> list[OutlierCluster]: ...
