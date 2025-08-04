from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.quality.outlier.prototype import BaseClusteringOperator, OutlierCluster

class HDBSCANClustering(BaseClusteringOperator):
    """Clustering using HDBSCAN."""
    outlier_col: Incomplete
    model: Incomplete
    def __init__(self, outlier_col=...) -> None: ...
    def fit(self, X: pdDataFrame): ...
    def predict(self, X: pdDataFrame, cluster_col: str = ...) -> list[OutlierCluster]: ...
