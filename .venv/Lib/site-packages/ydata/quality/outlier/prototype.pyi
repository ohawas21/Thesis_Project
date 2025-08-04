from _typeshed import Incomplete
from dataclasses import dataclass
from pandas import DataFrame as pdDataFrame, Index as pdIndex
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin
from ydata.dataset import Dataset

DEFAULT_IS_OUTLIER_SCORE_COL: str
DEFAULT_CLUSTER_COL: str

class BaseOperator:
    def summary(self) -> dict: ...

class BaseClusteringOperator(BaseOperator, ClusterMixin):
    outlier_col: Incomplete
    def __init__(self, outlier_col: str = ...) -> None: ...
    def fit(self, X: pdDataFrame): ...
    def fit_predict(self, X: pdDataFrame) -> pdDataFrame: ...

class BaseProjectionOperator(BaseOperator, TransformerMixin): ...
class BaseDetectionOperator(BaseOperator, BaseEstimator, ClassifierMixin): ...

class Identity(BaseOperator):
    """General Identidy Operator which can be used as default operator for any
    step in a pipeline."""
    def fit_transform(X: pdDataFrame, *args, **kwargs) -> pdDataFrame: ...
    def fit_predict(X: pdDataFrame, *args, **kwargs) -> pdDataFrame: ...
    def represent(X: pdDataFrame, *args, **kwargs) -> pdDataFrame: ...
    def transform(X: pdDataFrame, *args, **kwargs) -> pdDataFrame: ...

@dataclass
class OutlierSteps:
    '''The different steps of a Outlier Detection Pipeline.

    - projection: represent the dataset in a space that is suitable to detect outliers.
    - detection: the detection method used to find the outliers in the projected dataset.
    - clustering: the clustering technique to group the outliers detected by the `detection` step.
    - representation: a way to transform the dataset such that it is easy to explain or visualize outliers in 1D or 2D.

    Projection has to be understood has "transformation in a space that make the detection method
    explainable". For e.g.:
        - a detection in the space of 2D PCA means an outliers does not participate to explain the dataset variance
        - a detection in the space of std ration means an outliers is far from the mean
        - etc.
    '''
    projection: BaseProjectionOperator = ...
    detection: BaseDetectionOperator = ...
    clustering: BaseClusteringOperator = ...
    representation: BaseOperator = ...
    def __init__(self, projection=..., detection=..., clustering=..., representation=...) -> None: ...

@dataclass
class OutlierCluster:
    """Representation of a group of outliers sharing the same properties."""
    index: pdIndex
    pipeline: str = ...
    is_outlier: bool = ...
    def mark_inlier(self) -> None:
        """Mark a cluster as inliner, i.e. not an outlier.

        By feedback back this cluster to the engine, the pipeline is
        notified not to consider these points (and hopefully neighbors)
        as outliers in the future.
        """
    def get_outliers(self, X: Dataset | pdDataFrame) -> pdDataFrame:
        """Return the outliers elements from a dataset.

        Args:
            X (Union[Dataset, pdDataFrame]): Dataset containing the outliers

        Returns:
            pdDataFrame: DataFrame containins only the outliers
        """
    def __init__(self, index, pipeline=..., is_outlier=...) -> None: ...
