import importlib
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pandas import DataFrame as pdDataFrame

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.quality.exceptions import InvalidConfigurationError
from ydata.quality.outlier.pipeline import OutlierPipeline
from ydata.quality.outlier.prototype import OutlierCluster, OutlierSteps
from ydata.utils.configuration import TextStyle


class OutlierEngine:

    __default_config = os.path.join(
        os.path.dirname(__file__), "config.default.yaml")

    def __init__(self, config: Optional[Union[str, Path,  Dict[str, OutlierPipeline]]] = None):
        """Outlier Engine reponsible to find outliers in any dataset.

        Args:
            config (Optional[Union[str, Path,  Dict[str, OutlierPipeline]]]): Pipelines to use to detect outliers. It can be a path to a file or directly a dictionary of outliers pipelines.

        Returns:
            List[OutlierCluster]: List of outlier clusters
        """
        try:
            self.pipelines: Dict[str, OutlierPipeline] = self.__init_configuration(
                config)
        except Exception as e:
            raise InvalidConfigurationError(e) from e

        self._raw_clusters: Dict[str, OutlierCluster] = {}
        self.clusters = None
        self.dataset_len = None

    def __init_configuration(self, config: Optional[Union[str, Path,  Dict[str, OutlierPipeline]]] = None) -> Dict[str, OutlierPipeline]:
        """Initialize the configuration by building the pipelines from a YAML
        configuratio file.

        Args:
            config (Optional[Union[str, Path,  Dict[str, OutlierPipeline]]]): Pipelines to use to detect outliers. It can be a path to a file or directly a dictionary of outliers pipelines.

        Returns:
             Dict[str, OutlierPipeline]: List of outlier pipelines
        """
        if isinstance(config, dict):
            return config

        if config is None:
            config = OutlierEngine.__default_config

        with open(config) as f:
            data = yaml.safe_load(f)

        pipelines = {}
        for e in data['pipelines']:
            for pipeline_name, pipeline_info in e.items():
                p_config = {}
                for k, v in pipeline_info.items():
                    if isinstance(v, str):
                        p_config[k] = getattr(importlib.import_module(
                            f'ydata.quality.outlier.{k}'), v)()
                    else:
                        p_config[k] = getattr(importlib.import_module(
                            f'ydata.quality.outlier.{k}'), v['class'])(**v['params'])
                pipeline = OutlierPipeline(OutlierSteps(**p_config))
                pipelines[pipeline_name] = pipeline
        return pipelines

    def fit_predict(self, X: Dataset, metadata: Metadata) -> List[OutlierCluster]:
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
        self.dataset_len = X.nrows
        for pname, pp in self.pipelines.items():
            clusters = pp.fit_predict(X, metadata)
            for c in clusters:
                c.pipeline = pname
            self._raw_clusters[pname] = clusters

        clusters = list(chain(*self._raw_clusters.values()))
        clusters = self._rank_clusters(clusters)
        clusters = self._filter_clusters(clusters)
        clusters = self._merge_clusters(clusters)
        self.clusters = clusters
        return clusters

    def get_outlier_clusters(self) -> Dict[str, List[OutlierCluster]]:
        """Return the outlier clusters indexed on each pipeline without the
        engine ranking and post-processing.

        Returns:
            Dict[str, List[OutlierCluster]]: List of outlier clusters indexed by their pipeline
        """
        return self._raw_clusters

    def represent(self, X: Dataset, pipeline: Optional[str] = None) -> Union[Dict[str, pdDataFrame], pdDataFrame]:
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
        if pipeline:
            pp = self.pipelines.get(pipeline)
            if pp is None:
                raise KeyError(f"Pipeline {pipeline} does not exist!")
            return pp.represent(X)
        else:
            representations = {}
            for pname, pp in self.pipelines.items():
                representations[pname] = pp.represent(X)
            return representations

    def update(self,):
        """Update the pipelines with a user-feedback on the cluster or outlier
        elements."""
        raise NotImplementedError()

    def _rank_clusters(self, clusters: List[OutlierCluster]) -> List[OutlierCluster]:
        """Rank the cluster according to their importance in terms of size and
        score.

        Not implemented yet - several strategies are possible.

        Args:
            clusters (List[OutlierCluster]): List of clusters to rank.

        Returns:
            List[OutlierCluster]: Ranked list of clusters
        """
        return clusters

    def _filter_clusters(self, clusters: List[OutlierCluster]) -> List[OutlierCluster]:
        """Filter the cluster according to their importance in terms of size
        and score.

        Not implemented yet - several strategies are possible.

        Note(aquemy): Is it useful or should be let the user do the filtering knowing we provide the size and score?

        Args:
            clusters (List[OutlierCluster]): List of clusters to filter.

        Returns:
            List[OutlierCluster]: Filtered list of clusters
        """
        return clusters

    def _merge_clusters(self, clusters: List[OutlierCluster]) -> List[OutlierCluster]:
        """Merge the clusters according to their importance in terms of size
        and score.

        Not implemented yet - several strategies are possible.

        For instance, elements that are in several clusters (i.e. outliers for several metrics) might be
        more "outliers" than the elements appearing only in a single cluster. Therefore it might be interesting to
        regroup them into a "high ranked" cluster.

        Note(aquemy): It might be better to "recluster" or add more clusters with a high score rather than "merging" cluster.

        Args:
            clusters (List[OutlierCluster]): List of clusters to potentially merge.

        Returns:
            List[OutlierCluster]: List of clusters
        """
        return clusters

    def summary(self, details: bool = False) -> dict:
        """Generates a dictionary with a summary of the outlier engine.

        Returns:
            dict: The summary of the outlier engine.
        """
        total_num_outliers = len(set(chain(*[list(cluster.index) for cluster in self.clusters]))) \
            if self.clusters is not None else 0
        return {
            "total_num_pipelines": len(self.pipelines),
            "total_num_clusters": len(self.clusters) if self.clusters is not None else 0,
            "total_num_outliers": total_num_outliers,
            "total_num_outliers_per": total_num_outliers / self.dataset_len if self.dataset_len is not None else 0.0,
            "pipelines": [
                {
                    pname: pp.summary(details=details)
                    for pname, pp in self.pipelines.items()
                }
            ]
        }

    def __str__(self):
        """Returns a printable string representation of the outlier engine."""
        oe_summary = self.summary(details=False)
        str_repr = TextStyle.BOLD + "Outlier Engine Summary \n \n" + TextStyle.END
        str_repr += TextStyle.BOLD + "Total Number of Pipelines: " + TextStyle.END
        str_repr += str(oe_summary["total_num_pipelines"]) + "\n"
        str_repr += TextStyle.BOLD + "Total Number of Clusters: " + TextStyle.END
        str_repr += str(oe_summary["total_num_clusters"]) + "\n"
        str_repr += TextStyle.BOLD + "Total Number of Outliers: " + TextStyle.END
        str_repr += str(oe_summary["total_num_outliers"])
        str_repr += f' ({100 * oe_summary["total_num_outliers_per"]:.1f}% of the dataset) \n \n'
        str_repr += str("-" * 100) + "\n \n"
        for ix, (pip_name, pip) in enumerate(self.pipelines.items()):
            str_repr += TextStyle.BOLD + \
                f"[#{ix + 1}] {pip_name.capitalize()}" + TextStyle.END
            str_repr += " - " + str(pip) + "\n" + str("-" * 100) + "\n \n"
        return str_repr
