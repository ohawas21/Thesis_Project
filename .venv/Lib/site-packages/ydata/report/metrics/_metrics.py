from time import perf_counter

from numpy import abs as np_abs
from numpy import histogram as np_histogram
from numpy import histogram2d as np_histogram2d
from numpy import isnan
from numpy import log2 as np_log2
from numpy import mean as np_mean
from numpy import nonzero as np_nonzero
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from numpy import triu as np_triu
from numpy import zeros as np_zeros
from numpy.ma import masked_invalid
from numpy.random import choice as np_choice
from pandas import DataFrame as pdDataFrame
from scipy.special import expit
from statsmodels.tsa.stattools import acf

from ydata.report.logger import logger
from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.metrics.stat_tests import compute_cstest, compute_kstest
from ydata.report.styles.html import StyleHTML
from ydata.utils.associations import association_matrix
from ydata.utils.associations.measure_associations import MeasureAssociationsPandas
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType
from ydata.utils.metadata import get_types


class DistanceCorrelation(BaseMetric):
    """Calculate the correlation/strength-of-association of features in data-
    set with both categorical and continuous.

    features using: * Pearson's R for continuous-continuous cases * Correlation Ratio for categorical-continuous cases
    * Cramer's V or Theil's U for categorical-categorical cases.
    The function then computes the correlation score, which is the average of the difference between the elements of the
    the correlation matrices. The result will be a score between 0 and 1, with a score of 1 declaring that the
    correlation matrices are identical.
    """

    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @staticmethod
    def _get_description(formatter):
        description = f"The {formatter.bold('CORRELATION SIMILARITY')} measures how close \
            are synthetic and real correlation matrices. It is bounded between [0-1] metric, the \
            closer to 1, the higher fidelity."

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        return self.distance_correlation(source, synthetic, kwargs["metadata"])

    def distance_correlation(self, source, synthetic, metadata):
        # Calculate the correlation matrices.
        # Convert the correlation matrices to numpy arrays.
        datatypes = {k: v.datatype for k, v in metadata.columns.items(
        ) if k in source.columns and k in synthetic.columns}
        vartypes = {k: v.vartype for k, v in metadata.columns.items(
        ) if k in source.columns and k in synthetic.columns}

        corr_real = association_matrix(
            source, datatypes, vartypes, association_measurer=MeasureAssociationsPandas)
        corr_synth = association_matrix(
            synthetic, datatypes, vartypes, association_measurer=MeasureAssociationsPandas)

        # Get the nxn shape of the correlation matrices.
        n = corr_real.shape[1]

        # Calculate the size of the scores array, which stores the computed correlation differences.
        # ar_size is computed based on the quadratic sequence in the number of non-zero elements of an
        # upper triangular matrix excluding the main diagonal.
        ar_size = int(n * (n - 1) / 2)
        # Calculate the correlation differences.
        scores = np_triu(corr_real - corr_synth, 1)
        scores = np_sqrt(np_sum(scores * scores))

        # Calculate the correlation score, in which the average score is subtracted from 1, so the closer the score is to 1,
        # the more similar the matrices are.
        corr_score = 1 - (scores / ar_size)
        return corr_score

    @property
    def name(self) -> str:
        return "Correlation Similarity"


def distance_distribution(df_real, df_synth, data_types: dict):
    """Calculates how similar the distributions of the various columns are."""
    con_cols = []
    cat_cols = []
    for col_type in df_real.columns:
        if DataType(data_types[col_type]) == DataType.NUMERICAL:
            con_cols.append(col_type)
        elif DataType(data_types[col_type]) in CATEGORICAL_DTYPES:
            cat_cols.append(col_type)

    results = []
    for col in cat_cols:
        results.append(compute_cstest(df_real[col], df_synth[col]))
    categorical = np_mean(results)

    results = []
    for col in con_cols:
        results.append(compute_kstest(df_real[col], df_synth[col]))
    continuous = np_mean(results)

    if isnan(categorical):
        return continuous

    if isnan(continuous):
        return categorical

    return np_mean([continuous, categorical])


class DistanceDistribution(BaseMetric):
    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @staticmethod
    def _get_description(formatter):
        description = f"{formatter.bold('DISTANCE DISTRIBUTION')} measures the features' \
            distribution similarity between original and generated data. The Chi-squared \
            test evaluates features with discrete distributions, and the Kolmogorov-Smirnov \
            test evaluates features with continuous distributions. Returns values between \
            [0, 1], closer to 1 is desirable."

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        return distance_distribution(source, synthetic, kwargs["data_types"])

    @property
    def name(self) -> str:
        return "Distance Distribution"


class MutualInfo(BaseMetric):
    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @staticmethod
    def _get_description(formatter):
        description = f"{formatter.bold('MUTUAL INFORMATION')} measures how much \
            information can be obtained about one feature by observing another. \
            Unlike correlation and covariance, mutual information can measure \
            non-linear relationships. Returns values between [0, 1], closer to 1 is desirable."

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        return self._score(source, synthetic)

    @property
    def name(self) -> str:
        return "Mutual Information"

    def _calculate_mutual_information(self, feature_a, feature_b, bins: int):
        """Calculate mutual information between two features.

        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        discretized_a = np_histogram(feature_a, bins=bins)[0]
        discretized_b = np_histogram(feature_b, bins=bins)[0]
        discretized_ab = np_histogram2d(feature_a, feature_b, bins=bins)[0]

        return (
            self._entropy(discretized_a)
            + self._entropy(discretized_b)
            - self._entropy(discretized_ab)
        )

    def _entropy(self, feature):
        """Calculates Shannon entropy."""
        normalized = feature[np_nonzero(feature)]
        normalized = normalized / float(np_sum(normalized))

        return -np_sum(normalized * np_log2(normalized))

    def _mutual_information_score(self, data, bins=100):
        columns = data.columns
        upper_matrix_size = int(len(columns) * (len(columns) - 1) / 2)
        scores = np_zeros(upper_matrix_size)
        idx = 0
        for i, feat_a in enumerate(columns):
            for feat_b in columns[i + 1:]:
                scores[idx] = self._calculate_mutual_information(
                    data[feat_a], data[feat_b], bins
                )
                idx += 1
        return expit(scores)

    def _score(self, source, synthetic):
        source_score = self._mutual_information_score(source)
        synth_score = self._mutual_information_score(synthetic)

        # after appling the sigmoid, since all values are positive
        # the max difference will be 0.5, we multiply by 2 to rescale
        diff = np_abs(synth_score - source_score) * 2
        return 1 - np_mean(diff)


class Autocorrelation(BaseMetric):
    def __init__(self, formatter=StyleHTML, exclude_entity_col: bool = True) -> None:
        super().__init__(formatter,  exclude_entity_col)

    @staticmethod
    def _get_description(formatter):
        description = (
            f"The {formatter.bold('AUTOCORRELATION')} provides the average "
            + "autocorrelations within features of dataset."
        )

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        return self._autocorrelation(
            source, synthetic, kwargs["entity_data"], kwargs["data_types"]
        )

    def _entity_aware_autocorrelation(
        self,
        dataframe: pdDataFrame,
        entity_data: pdDataFrame,
        data_types,
        max_entities: int = 5000,
        max_seconds: int = 120,
    ):
        entities_df = entity_data.apply(lambda x: tuple(x), axis=1)
        entities = entities_df.unique()
        if len(entities) > max_entities:
            entities_idx = np_choice(
                list(range(len(entities))), max_entities, replace=False
            )
            entities = [entities[i] for i in entities_idx]
            # filter the dataset to reduce the cost of the following queries
            mask = entities_df.isin(entities)
            dataframe = dataframe[mask]

        scores = []
        start = perf_counter()
        for i, entity in enumerate(entities):
            entity_filtered_data = entity_data[(
                entity_data == entity).all(axis=1)]
            df = dataframe.loc[entity_filtered_data.index]
            scores.append(self._calc_autocorrelation_score(df, data_types))

            if perf_counter() - start > max_seconds:
                logger.info(
                    f"[PROFILEREPORT] - Autocorrelation excedded allocated time of {max_seconds}s.)"
                )
                logger.info(
                    f"[PROFILEREPORT] - Autocorrelation calculated for {i + 1} entitiess."
                )
                break

        return masked_invalid(scores).mean()

    def _calc_autocorrelation_score(self, dataframe: pdDataFrame, data_types, nlags=50):
        scores = []
        for k, v in data_types.items():
            if v == DataType.NUMERICAL:
                try:
                    scores.append(np_mean(acf(dataframe[k], nlags=nlags)))
                except BaseException:
                    continue

        return masked_invalid(scores).mean()

    def _autocorrelation(
        self,
        df_real: pdDataFrame,
        df_synth: pdDataFrame,
        entity_data: dict,
        data_types=None,
    ):
        "Computes the average autocorrelation between features of real and synthetic datasets."
        if data_types is None:
            data_types = get_types(df_real)

        if not entity_data["real"].empty:
            real_score = self._entity_aware_autocorrelation(
                df_real, entity_data["real"], data_types
            )
            synth_score = self._entity_aware_autocorrelation(
                df_synth, entity_data["synth"], data_types
            )
        else:
            real_score = self._calc_autocorrelation_score(df_real, data_types)
            synth_score = self._calc_autocorrelation_score(
                df_synth, data_types)

        return {
            "real_score": real_score,
            "synth_score": synth_score
        }

    @property
    def name(self) -> str:
        return "Autocorrelation"
