from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from pandas import DataFrame as pdDataFrame
from pandas import Index as pdIndex

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.quality.outlier.prototype import DEFAULT_IS_OUTLIER_SCORE_COL, Identity, OutlierCluster, OutlierSteps
from ydata.utils.configuration import TextStyle


class OutlierPipeline:
    def __init__(self, steps: Union[Dict, OutlierSteps]):
        super().__init__()
        self.steps = steps if isinstance(
            steps, OutlierSteps) else OutlierSteps(**steps)
        self.outlier_score_ = None
        self.cluster_index_ = {}
        self.dataset_len = None
        self.columns = None

    def _get_outliers(self, X: pdDataFrame, outlier_col: str, threshold: Optional[float] = None) -> pdDataFrame:
        """Determine the cutoff threshold to consider a point an outlier. By
        default and by convention, a negative score indicates non-outliers and
        (strictly) positive score indicates outliers. However, if the detection
        method implement a field `threshold` it will be used. This can be used
        in case the method has a different convention or simply to tune the
        threshold automatically from outside.

        Args:
            X (pdDataFrame): Dataframe
            outlier_col (str): name for the column that will contain the outlier score
            threshold: threshold to use to determine an outlier

        Returns:
            pdDataFrame: The outliers with a column indicating their score
        """
        if threshold is None:
            threshold = self.steps.detection.threshold if hasattr(
                self.steps.detection, 'threshold') else 0.0
        X[outlier_col] = self.outlier_score_
        outliers = X[X[outlier_col] > threshold]
        return outliers

    def fit_predict(self, X: Dataset, metadata: Metadata, outlier_col: str = DEFAULT_IS_OUTLIER_SCORE_COL) -> List[OutlierCluster]:
        """Train the outlier pipeline and find the outliers in a dataset.

        Args:
            X (Dataset): Dataset containing the outliers
            metadata (Metadata): Metadata associated to the dataset
            outlier_col (str): name for the column that will contain the outlier score

        Returns:
            pdDataFrame: The outliers with a column indicating their score
        """
        X_ = X.to_pandas()  # Right now we assume that there is no Dask version
        self.dataset_len = len(X_)

        # For now the pipelines accept only numerical columns
        self.columns = metadata.numerical_vars
        X_ = X_[self.columns]

        X_p = self.steps.projection.fit_transform(X_)
        self.outlier_score_ = pdDataFrame(self.steps.detection.fit_predict(
            X_p), columns=[DEFAULT_IS_OUTLIER_SCORE_COL])
        X_p = self._get_outliers(X_p, outlier_col)
        if X_p.shape[0] > 1:
            X_f = self.steps.clustering.fit_predict(X_p)
            for i, c in enumerate(X_f):
                self.cluster_index_[i] = c.index
        elif X_p.shape[0] == 1:
            y = X_p.index
            X_f = list(map(lambda i: OutlierCluster(index=pdIndex([i])), y.values))
        else:
            X_f = []
        return X_f

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
        X_ = X.to_pandas()  # Right now we assume that there is no Dask version
        X_ = X_[self.columns]

        # First, we project the dataset into the space in which the outliers are detected
        X_p = self.steps.projection.transform(X_)
        # If the pipeline provides a special representation, we need to prepare the data here accordingly
        X_r = self.steps.representation.transform(X_p)

        X_r['outlier_score'] = self.outlier_score_
        X_r['cluster'] = None  # No one is guilty until proven otherwise :)
        for i, index in self.cluster_index_.items():
            X_r.loc[index, "cluster"] = i
        return X_r

    @staticmethod
    def _set_visibility(
        axis: Any, tick_mark: str = "none"
    ) -> Any:
        for anchor in ["top", "right", "bottom", "left"]:
            axis.spines[anchor].set_visible(False)
        axis.xaxis.set_ticks_position(tick_mark)
        axis.yaxis.set_ticks_position(tick_mark)
        return axis

    def plot(self, X: Dataset, ax=None, **kwargs):
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
        rep = self.represent(X)
        if self.steps.representation != Identity:
            return self.steps.representation.plot(rep, ax)
        else:  # Default 2D plot of the representation
            non_outliers = rep[rep['cluster'].isna()]
            outliers_df = rep.dropna()
            if ax is None:
                _, ax = plt.subplots(figsize=[12, 5])

            # Should colors an the topk go to config?
            colors = ["#909296", "#0052CC", "#408DFF",
                      "#BD0F06", "#F04F3A", "#700002"]
            ax.scatter(
                x=rep.loc[non_outliers.index, [0]],
                y=rep.loc[non_outliers.index, [1]],
                color=colors[0],
                s=10,
                alpha=0.5,
                marker=".",
                label="non-outliers"
            )

            topk = list(outliers_df.cluster.value_counts().head(4).index)

            for idx, cluster in enumerate(topk):
                cluster_df = outliers_df[outliers_df.cluster == cluster]
                ax.scatter(
                    x=rep.iloc[cluster_df.index, [0]],
                    y=rep.iloc[cluster_df.index, [1]],
                    color=colors[idx + 1],
                    s=100 * cluster_df.outlier_score,
                    alpha=0.5,
                    label=f"Cluster_id {cluster}"
                )

            cluster_df = outliers_df[~outliers_df.cluster.isin(topk)]
            if len(cluster_df) > 0:
                ax.scatter(
                    x=rep.iloc[cluster_df.index, [0]],
                    y=rep.iloc[cluster_df.index, [1]],
                    color=colors[-1],
                    s=100 * cluster_df.outlier_score,
                    alpha=0.5,
                    label="Other clusters"
                )

            # Put a legend to the right of the current axis
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend(frameon=False, loc="center left",
                               bbox_to_anchor=(1, 0.5))
            for legend_handle in legend.legend_handles:
                legend_handle._sizes = [30]

            ax = self._set_visibility(ax)
            return ax

    def summary(self, details: bool = False) -> dict:
        """Generates a dictionary with a summary of the pipeline.

        Returns:
            dict: The summary of the pipeline.
        """
        if details:
            identity_summary = {"name": Identity.__name__}
            pip_summary = {
                "steps": {
                    "projection": self.steps.projection.summary()
                    if self.steps.projection != Identity else identity_summary,
                    "detection": self.steps.detection.summary()
                    if self.steps.detection != Identity else identity_summary,
                    "clustering": self.steps.clustering.summary()
                    if self.steps.clustering != Identity else identity_summary,
                    "representation": self.steps.representation.summary()
                    if self.steps.representation != Identity else identity_summary
                }
            }
        else:
            pip_summary = {}

        if self.outlier_score_ is not None:
            total_num_outliers = sum([len(cluster)
                                     for cluster in self.cluster_index_.values()])
            pip_summary["total_num_clusters"] = len(self.cluster_index_)
            pip_summary["total_num_outliers"] = total_num_outliers
            pip_summary["total_num_outliers_per"] = total_num_outliers / \
                self.dataset_len

            pip_summary["clusters"] = pdDataFrame([
                {
                    "cluster_id": id,
                    "num_outliers": len(indexes),
                    "outlier_score_mean": self.outlier_score_.loc[indexes].values.mean(),
                    "outlier_score_std": self.outlier_score_.loc[indexes].values.std(),
                    "outlier_score_min": self.outlier_score_.loc[indexes].values.min(),
                    "outlier_score_max": self.outlier_score_.loc[indexes].values.max()
                }
                for id, indexes in self.cluster_index_.items()
            ])
        else:
            pip_summary["total_num_clusters"] = 0
            pip_summary["total_num_outliers"] = 0
            pip_summary["total_num_outliers_per"] = 0.0
            pip_summary["clusters"] = None

        return pip_summary

    def __str__(self):
        """Returns a printable string representation of the pipeline."""
        pip_summary = self.summary(details=False)
        str_repr = TextStyle.BOLD + "Outlier Pipeline Summary \n \n" + TextStyle.END
        if pip_summary["clusters"] is not None:
            str_repr += TextStyle.BOLD + "Total Number of Clusters: " + TextStyle.END
            str_repr += str(pip_summary["total_num_clusters"]) + "\n"
            str_repr += TextStyle.BOLD + "Total Number of Outliers: " + TextStyle.END
            str_repr += str(pip_summary["total_num_outliers"])
            str_repr += f' ({100 * pip_summary["total_num_outliers_per"]:.1f}% of the dataset) \n \n'
            clusters = pip_summary["clusters"].reset_index(drop=True)
            clusters = clusters.rename({
                "cluster_id": "ID",
                "num_outliers": "Number of Outliers",
                "outlier_score_mean": "Outlier Score Mean",
                "outlier_score_std": "Outlier Score STD",
                "outlier_score_min": "Outlier Score Min",
                "outlier_score_max": "Outlier Score Max",
            }, axis='columns')
            str_repr += TextStyle.BOLD + \
                "Clusters (from the latest execution)" + TextStyle.END + "\n"
            str_repr += clusters.to_string(index=False,
                                           float_format=lambda x: "{:.3f}".format(x)) + "\n"
        else:
            str_repr += TextStyle.BOLD + \
                "The pipeline was not yet executed." + TextStyle.END + "\n"
        return str_repr
