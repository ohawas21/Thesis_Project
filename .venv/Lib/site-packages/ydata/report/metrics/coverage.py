import pandas as pd

from ydata.report.metrics.top_bottom_metric import TopBottomMetric
from ydata.report.metrics.utils import get_categorical_vars, get_numerical_vars
from ydata.report.styles.html import StyleHTML


class CategoryCoverage(TopBottomMetric):
    """Category Coverage Metric.

    Compute the fraction of real data categories that are also
    represented in the synthetic data. Provide the ratio of
    representation of the category in the real and in the synthetic
    dataset.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @property
    def name(self) -> str:
        return "Category Coverage"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('CATEGORY COVERAGE (CC)')} computes the \
            ratio of real data categories that are represented in the synthetic \
            data. The two tables below present the five features with the highest \
            and lowest coverage."

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the category coverage metric.

        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.
        Returns: a dict with the top and bottom category coverage ratios per categorical column.
        """
        categorical_vars = get_categorical_vars(source, kwargs["metadata"])
        results = {}
        for col in categorical_vars:
            synth_values = [sv for sv in synthetic[col].dropna(
            ).unique() if sv in source[col].dropna().unique()]
            real_nunique = source[col].dropna().nunique()
            synth_nunique = len(synth_values)
            results[col] = synth_nunique / real_nunique

        return self._get_results(results=results, name="CC")


class MissingCategoryCoverage(TopBottomMetric):
    """Missing Category Coverage Metric.

    Compute the categorical values from the real data that are not
    represented in the synthetic data. Provide the categories that have
    not been represented.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @property
    def name(self) -> str:
        return "Missing Category Coverage"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('MISSING CATEGORY COVERAGE (MCC)')} computes \
            the similarity ratio of the cardinality between the real and synthetic \
            categorical values. The two tables below present the five features with \
            the highest and lowest coverage."

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the category coverage missing metric.

        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.
        Returns: a dict with the top and bottom category coverage missing ratios per categorical column.
        """
        categorical_vars = get_categorical_vars(source, kwargs["metadata"])
        real = source[categorical_vars].copy()
        synth = synthetic[categorical_vars].copy()
        results = {}
        for col in categorical_vars:
            real_val_counts = real[col].dropna().value_counts()
            synth_val_counts = synth[col].dropna().value_counts()
            sim_col = 0.0
            for val, real_count in real_val_counts.items():
                if val in synth_val_counts:
                    sim_col += 1.0 - \
                        (abs(synth_val_counts[val] - real_count) /
                         max(real_count, synth_val_counts[val]))
                else:
                    # If the value doesn't exist in the synthetic data, the similarity is 0.
                    sim_col += 0.0
            results[col] = sim_col / real[col].nunique()

        return self._get_results(results=results, name="MCC")


class RangeCoverage(TopBottomMetric):
    """Range Coverage Metric.

    Compute the numerical variables ranges found in the real dataset are
    also represented in the synthetic.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @property
    def name(self) -> str:
        return "Range Coverage"

    @staticmethod
    def _get_description(formatter):
        return f"The {formatter.bold('RANGE COVERAGE (RC)')} computes the similarity ratio \
            between the numerical variables domain in the real dataset compared to the synthetic \
            one. The two tables below present the five features with the highest and lowest coverage."

    def get_minmax(self, dataset: pd.DataFrame):
        """Calculates the min-max stats of a provided dataframe."""
        stats = dict.fromkeys(['min', 'max'], {})
        stats['min'] = dataset.min()
        stats['max'] = dataset.max()
        return stats

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):
        """Computes the range coverage metric.

        Args:
            source (DataFrame): the original dataset.
            synthetic (DataFrame): the synthetic dataset.
        Returns: a dict with the top and bottom range coverage scores for all the numerical variables.
        """
        numerical_vars = get_numerical_vars(source, kwargs["metadata"])
        real = source[numerical_vars].copy()
        synth = synthetic[numerical_vars].copy()

        minmax_stats = {}
        minmax_stats['real'] = self.get_minmax(dataset=real)
        minmax_stats['synth'] = self.get_minmax(dataset=synth)

        results = {}

        for col in numerical_vars:
            max_r = minmax_stats['real']['max'][col]
            min_r = minmax_stats['real']['min'][col]

            if max_r == min_r:
                results[col] = 1.0
                continue

            normalized_min = max(
                (minmax_stats['synth']['min'][col] - min_r) / (max_r - min_r), 0)
            normalized_max = max(
                (max_r - minmax_stats['synth']['max'][col]) / (max_r - min_r), 0)
            results[col] = max(1 - (normalized_min + normalized_max), 0)

        return self._get_results(results=results, name="RC")
