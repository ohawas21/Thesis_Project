from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.dataset import Dataset as Dataset
from ydata.metadata import Metadata as Metadata
from ydata.quality.outlier.prototype import OutlierCluster, OutlierSteps

class OutlierPipeline:
    steps: Incomplete
    outlier_score_: Incomplete
    cluster_index_: Incomplete
    dataset_len: Incomplete
    columns: Incomplete
    def __init__(self, steps: dict | OutlierSteps) -> None: ...
    def fit_predict(self, X: Dataset, metadata: Metadata, outlier_col: str = ...) -> list[OutlierCluster]:
        """Train the outlier pipeline and find the outliers in a dataset.

        Args:
            X (Dataset): Dataset containing the outliers
            metadata (Metadata): Metadata associated to the dataset
            outlier_col (str): name for the column that will contain the outlier score

        Returns:
            pdDataFrame: The outliers with a column indicating their score
        """
    def represent(self, X: Dataset) -> pdDataFrame:
        """Represent a dataset according to the pipeline such that the outliers
        can be easily explained or visualized. In particular, a representation
        always contains two additional columns: `outlier_score` and `cluster`.
        The first indicates the confidence for a data point to be an outlier
        while the cluster indicates in which cluster a particular point belongs
        to. If the point is not an outlier, `cluster` value is `None`.

        Args:
            X (Dataset): Dataset to represent in the space of outliers.

        Returns:
            pdDataFrame: The dataset representation according to the pipeline.
        """
    def plot(self, X: Dataset, ax: Incomplete | None = None, **kwargs):
        """Plot the dataset in such a way that it ease the understanding of why
        a point is an outlier. The plot can be configured via the optional `ax`
        argument (a matplotlib Axes or AxesSubplot object).

        Because 'easy to understand' is specific to the pipeline, we let the pipeline superseeds the default plot via
        the step `representation`.
        However, because a plot must be 1D or 2D, we provide default visualization for such cases. A HUGE assumption
        done for the default plot is that the columns are sorted by importance. It is the case for most projection objects
        such as PCA, ICA or any other technique but for more *custom* representation it should be taken into account.

        note(aquemy): only the 2D plots is implemented right now. Not sure how to integrate a 1D plot. Maybe a flag at the OutlierSteps definition?

        Args:
            X (Dataset): Dataset to represent in the space of outliers.

        Returns:
            pdDataFrame: The dataset representation according to the pipeline.
        """
    def summary(self, details: bool = False) -> dict:
        """Generates a dictionary with a summary of the pipeline.

        Returns:
            dict: The summary of the pipeline.
        """
