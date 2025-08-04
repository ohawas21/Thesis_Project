import numpy as np
import pandas as pd

from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.metrics.utils import get_numerical_vars
from ydata.report.styles.html import StyleHTML


class StatisticalSimilarity(BaseMetric):
    """Statistical similarity metric.

    This metric is only computed for numerical variables.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)
        self._f_stats = {
            'mean': pd.DataFrame.mean,
            'std': pd.DataFrame.std,
            'median': (pd.DataFrame.quantile, 0.5),
            'q25': (pd.DataFrame.quantile, 0.25),
            'q75': (pd.DataFrame.quantile, 0.75)
        }
        self._labels = {
            'mean': "Mean",
            'std': "Std. Dev.",
            'median': "Median",
            'q25': "Q25%",
            'q75': "Q75%"
        }

    @property
    def name(self) -> str:
        return "Statistical Similarity"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('STATISTICAL SIMILARITY')} measures \
            how similar are the synthetic and real data considering five metrics: \
            mean, standard deviation, median, 25% quantile, and 75% quantile. \
            Each similarity is bounded between [0-1], where 1 represents equal \
            values. Only numerical features are considered in this analysis."

    def _compute(self, real, min_r, max_r, dataset):
        """Computes the statistical similarity of a given dataset against the
        real data stats."""
        results = {}
        support = (max_r - min_r).replace(0, 1)
        for key, func in self._f_stats.items():
            if key in ['median', 'q25', 'q75']:
                quant = func[1]
                func = func[0]
                results[key] = np.exp(
                    -abs(func(real, quant) - func(dataset, quant)) / support
                )
            else:
                results[key] = np.exp(
                    -abs(func(real) - func(dataset)) / support
                )
        return results

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the statistical similarity metric.

        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.
        Returns: a DataFrame with the statistical similarity metric.
        """
        numerical_vars = get_numerical_vars(source, kwargs["metadata"])
        real = source[numerical_vars].copy().dropna()
        synth = synthetic[numerical_vars].copy().dropna()
        results = self._compute(real=real, min_r=real.min(),
                                max_r=real.max(), dataset=synth)
        results_mean = {
            self._labels[k]: np.mean(v)
            for k, v in results.items()
        }
        results_df = pd.DataFrame.from_dict(results).round(2)
        results_df = results_df.rename(columns=self._labels)
        results_df.insert(loc=0, column='Feature', value=real.columns)
        return {
            "mean": results_mean,
            "dataframe": results_df
        }
