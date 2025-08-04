from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from pathlib import Path
from ydata.dataset import Dataset as Dataset
from ydata.metadata import Metadata as Metadata
from ydata.quality.outlier.pipeline import OutlierPipeline
from ydata.quality.outlier.prototype import OutlierCluster

class OutlierEngine:
    pipelines: Incomplete
    clusters: Incomplete
    dataset_len: Incomplete
    def __init__(self, config: str | Path | dict[str, OutlierPipeline] | None = None) -> None:
        """Outlier Engine reponsible to find outliers in any dataset.

        Args:
            config (Optional[Union[str, Path,  Dict[str, OutlierPipeline]]]): Pipelines to use to detect outliers. It can be a path to a file or directly a dictionary of outliers pipelines.

        Returns:
            List[OutlierCluster]: List of outlier clusters
        """
    def fit_predict(self, X: Dataset, metadata: Metadata) -> list[OutlierCluster]:
        """Train the registered outlier detection pipelines and find the
        outliers.

        Note(aquemy): For now only the `fit_transform` is implemented but in the future we could have `predict` such that
                      we can do novelty detection but also have a posterior check on the synthesizer by checking that it does not
                      introduce outliers. [SD-595]

        Args:
            X (Dataset): Dataset in which to detect outliers
            metadata (Metadata): Associated metadata object

        Returns:
            List[OutlierCluster]: List of outlier clusters
        """
    def get_outlier_clusters(self) -> dict[str, list[OutlierCluster]]:
        """Return the outlier clusters indexed on each pipeline without the
        engine ranking and post-processing.

        Returns:
            Dict[str, List[OutlierCluster]]: List of outlier clusters indexed by their pipeline
        """
    def represent(self, X: Dataset, pipeline: str | None = None) -> dict[str, pdDataFrame] | pdDataFrame:
        """Represent a dataset according to the pipelines such that the
        outliers can be easily explained or visualized. In particular, a
        representation always contains two additional columns: `outlier_score`
        and `cluster`. The first indicates the confidence for a data point to
        be an outlier while the cluster indicates in which cluster a particular
        point belongs to. If the point is not an outlier, `cluster` value is
        `None`.

        Args:
            X (Dataset): Dataset to represent in the space of outliers.
            pipeline (Optional[str]): Represent only for the specified pipeline. If None, return a dictionary with the representation for all pipelines.

        Returns:
            Union[Dict[str, pdDataFrame], pdDataFrame]: A dictionary with all representation if `pipeline` is None, otherwise the specific representation of `pipeline`
        """
    def update(self) -> None:
        """Update the pipelines with a user-feedback on the cluster or outlier
        elements."""
    def summary(self, details: bool = False) -> dict:
        """Generates a dictionary with a summary of the outlier engine.

        Returns:
            dict: The summary of the outlier engine.
        """
