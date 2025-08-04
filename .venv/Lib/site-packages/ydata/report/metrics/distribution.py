import pandas as pd
from numpy import isnan
from scipy.stats import kstest

from ydata.report.metrics.top_bottom_metric import TopBottomMetric
from ydata.report.metrics.utils import get_categorical_vars, get_numerical_vars
from ydata.report.styles.html import StyleHTML


class KSTest(TopBottomMetric):
    """Kolmogorov-Smirnov test based metric for numerical columns.

    This metric uses the Kolmogorov-Smirnov test to compare the
    distribution between two continuous columns (real and synth) using
    the empirical CDF. The test returns the resulting p-value. A small
    value means that we can reject the null hypothesis, meaning, the
    distributions are most probably different.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @property
    def name(self) -> str:
        return "Kolmogorov-Smirnov Test"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('KOLMOGOROV-SMIRNOV (KS) TEST')} compares the distribution \
            between two continuous variables (real and synthetic data) using the empirical CDF. \
            The two tables below present the five features with the highest and lowest values for this test."

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the Kolmogorov-Smirnov test based metric.

        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.
        Returns: a dict with the top and bottom Kolmogorov-Smirnov test metrics per numerical column.
        """
        numerical_vars = get_numerical_vars(source, kwargs["metadata"])
        real = source[numerical_vars].copy()
        synth = synthetic[numerical_vars].copy()
        results = {}
        for col in numerical_vars:
            results[col] = 1.0 - \
                kstest(real[col].dropna(), synth[col].dropna()).statistic

        return self._get_results(results=results, name="KS Test")


class TotalVariationDistance(TopBottomMetric):
    """Total Variation Distance metric for categorical columns.

    This metric computes the Total Variation Distance to compare the
    marginal distribution between two categorical columns (real and
    synth). The metric returns the complement of the Total Variation
    Distance so that higher values represent more similar distributions.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @property
    def name(self) -> str:
        return "Total Variation Distance"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('Total Variation Distance (TVD)')} compares the marginal \
            distribution between two categorical variables (real and synthetic data). The \
            two tables below present the five features with the highest and lowest values \
            for this metric."

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the Total Variation Distance metric.

        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.
        Returns: a dict with the top and bottom Total Variation Distance metrics per categorical column.
        """
        categorical_vars = get_categorical_vars(source, kwargs["metadata"])
        real = source[categorical_vars].copy()
        synth = synthetic[categorical_vars].copy()
        results = {}
        for col in categorical_vars:
            if real[col].dropna().nunique() > 1:
                real_val_counts = real[col].dropna().value_counts()
                synth_val_counts = synth[col].dropna().value_counts()
                for rv in real_val_counts.index:
                    if rv not in synth_val_counts.index:
                        synth_val_counts.at[rv] = 0
                drop_unexp_vals = [
                    sv for sv in synth_val_counts.index if sv not in real_val_counts.index]
                synth_val_counts = synth_val_counts.drop(
                    labels=drop_unexp_vals)
                f_exp = real_val_counts / real_val_counts.sum()
                f_obs = synth_val_counts / synth_val_counts.sum()
                total_variation = 0.0
                for cat in real_val_counts.index:
                    total_variation += abs(f_obs[cat] - f_exp[cat])
                if not isnan(total_variation):
                    results[col] = 1 - 0.5 * total_variation
            else:
                # If the variable has a single value, the result is assumed to be 1.
                results[col] = 1.0

        return self._get_results(results=results, name="TVD")
