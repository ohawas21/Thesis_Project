from pandas import DataFrame as pdDataFrame
from ydata.quality.outlier.prototype import BaseClusteringOperator, OutlierCluster

class IndividualClustering(BaseClusteringOperator):
    """Naive clustering which return a cluster per outlier elements."""
    def predict(self, X: pdDataFrame) -> list[OutlierCluster]: ...
