from dataclasses import dataclass
from typing import Union

from pandas import DataFrame as pdDataFrame
from pandas import Index as pdIndex
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin

from ydata.dataset import Dataset

DEFAULT_IS_OUTLIER_SCORE_COL = '_is_outlier_score'
DEFAULT_CLUSTER_COL = '_cluster'


class BaseOperator:

    def summary(self) -> dict:
        return {"name": self.__class__.__name__}


class BaseClusteringOperator(BaseOperator, ClusterMixin):

    def __init__(self, outlier_col: str = DEFAULT_IS_OUTLIER_SCORE_COL):
        self.outlier_col = outlier_col

    def fit(self, X: pdDataFrame):
        return self

    def fit_predict(self, X: pdDataFrame) -> pdDataFrame:
        self = self.fit(X)
        X = self.predict(X)
        return X


class BaseProjectionOperator(BaseOperator, TransformerMixin):
    pass


class BaseDetectionOperator(BaseOperator, BaseEstimator, ClassifierMixin):
    pass


class Identity(BaseOperator):
    """General Identidy Operator which can be used as default operator for any
    step in a pipeline."""

    def fit_transform(X: pdDataFrame, *args, **kwargs) -> pdDataFrame:
        return X

    def fit_predict(X: pdDataFrame, *args, **kwargs) -> pdDataFrame:
        return X

    def represent(X: pdDataFrame, *args, **kwargs) -> pdDataFrame:
        return X

    def transform(X: pdDataFrame, *args, **kwargs) -> pdDataFrame:
        return X


@dataclass
class OutlierSteps:
    """The different steps of a Outlier Detection Pipeline.

    - projection: represent the dataset in a space that is suitable to detect outliers.
    - detection: the detection method used to find the outliers in the projected dataset.
    - clustering: the clustering technique to group the outliers detected by the `detection` step.
    - representation: a way to transform the dataset such that it is easy to explain or visualize outliers in 1D or 2D.

    Projection has to be understood has "transformation in a space that make the detection method
    explainable". For e.g.:
        - a detection in the space of 2D PCA means an outliers does not participate to explain the dataset variance
        - a detection in the space of std ration means an outliers is far from the mean
        - etc.
    """
    projection: BaseProjectionOperator = Identity
    detection: BaseDetectionOperator = Identity
    clustering: BaseClusteringOperator = Identity
    representation: BaseOperator = Identity


@dataclass
class OutlierCluster:
    """Representation of a group of outliers sharing the same properties."""
    index: pdIndex
    pipeline: str = None
    is_outlier: bool = True

    def mark_inlier(self):
        """Mark a cluster as inliner, i.e. not an outlier.

        By feedback back this cluster to the engine, the pipeline is
        notified not to consider these points (and hopefully neighbors)
        as outliers in the future.
        """
        self.is_outlier = False

    def get_outliers(self, X: Union[Dataset, pdDataFrame]) -> pdDataFrame:
        """Return the outliers elements from a dataset.

        Args:
            X (Union[Dataset, pdDataFrame]): Dataset containing the outliers

        Returns:
            pdDataFrame: DataFrame containins only the outliers
        """
        # Dask does not support positional indexing
        # A way to improve this would be to work with mask only
        if isinstance(X, Dataset):
            return X.to_pandas().iloc[self.index]
        else:
            return X.iloc[self.index]
