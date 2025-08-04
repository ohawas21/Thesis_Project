"""
    Synthesizers preprocess methods
"""
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import logging
import sys
from copy import deepcopy
from decimal import Context, Decimal
from random import randint
from typing import Union, Dict

import numpy as np
import pandas as pd
from dask import compute
from dask.dataframe import DataFrame as ddDataFrame
from dask.dataframe import concat, to_datetime

from numpy import array as nparray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.utils._testing import ignore_warnings

from ydata.dataset import Dataset
from ydata.metadata.column import Column
from ydata.metadata.metadata import Metadata
from ydata.metadata.utils import drop_null
from ydata.preprocessors.base import BaseOperator
from ydata.preprocessors.exceptions import MaxIteration
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder, AnonymizerType
from ydata.preprocessors.methods.encoders import OrdinalEncoder
from ydata.preprocessors.methods.gaussianization.rbig.config import RBIGConfig
from ydata.preprocessors.methods.gaussianization.rbig.model import RBIG
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType, VariableType
from ydata.utils.misc import log_time_factory

from ydata.utils.logger import SDKLogger

metrics_logger = SDKLogger(name="Metrics logger")

def synthlogger_config(verbose):
    logger = logging.getLogger("Synthesizer")
    log_file = os.getenv("LOG_FILE")

    if log_file:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            filename=os.getenv("SYNTH_LOG_PATH"),
        )
    else:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            stream=sys.stdout,
        )

    return logger


logger = synthlogger_config(verbose=os.getenv(
    "VERBOSE", "false").lower() == "true")


# FIXME unused code
class OneHotEncoderTopN(OneHotEncoder):
    def __init__(self, *, max_categories: Union[int, float] = 10):
        super().__init__(sparse=False, handle_unknown="ignore")

        if isinstance(max_categories, float):
            assert 0.0 < max_categories <= 1.0
        elif isinstance(max_categories, int):
            assert 0 < max_categories
        else:
            raise TypeError

        self.max_categories = max_categories
        self.categories = []
        self.grouped_categories = []

    @log_time_factory(logger)
    def fit(self, X, y=None):
        X = np.copy(X)
        n = self.max_categories

        for i in range(X.shape[1]):
            values, counts = np.unique(X[:, i], return_counts=True)
            sort_idx = np.argsort(counts)[::-1]
            values, counts = values[sort_idx], counts[sort_idx]

            if isinstance(n, float):
                n = max(1, np.argmax(np.cumsum(counts) / X.shape[0] >= n))

            self.categories.append(values[:n])
            self.grouped_categories.append(
                dict(zip(values[n:], counts[n:] / np.sum(counts[n:])))
            )

        return super().fit(X, y)

    @log_time_factory(logger)
    def inverse_transform(self, X):
        X = super().inverse_transform(X)

        for i in range(X.shape[1]):
            if np.any(X[:, i] == None) and self.grouped_categories[i]:  # noqa: E711
                X[X[:, i] == None, i] = np.random.choice(  # noqa: E711
                    list(self.grouped_categories[i].keys()),
                    size=np.sum(X[:, i] == None),  # noqa: E711
                    p=list(self.grouped_categories[i].values()),
                )
        return X


class MultimodalTransformer(GaussianMixture, TransformerMixin):
    def __init__(self, *, n_components=10):
        super().__init__(n_components=n_components)

    @log_time_factory(logger)
    def transform(self, X):
        X = np.array(X).copy()

        labels = super().predict(X)
        k = len(self.weights_)

        result = []
        for i in range(X.shape[1]):
            Xc = np.tile(X[:, [i]], [1, k])

            for c in range(k):
                Xc[labels != c, c] = 0

            result.append(Xc)

        return np.concatenate(result, axis=1)

    @log_time_factory(logger)
    def inverse_transform(self, X):
        k = len(self.weights_)
        result = np.stack(
            [np.sum(X[:, i * k: (i + 1) * k], axis=1)
             for i in range(X.shape[1] // k)],
            axis=1,
        )

        return result

    def get_feature_names(self, input_features=None):
        raise NotImplementedError


class ClipTransformer(BaseEstimator, TransformerMixin):
    @log_time_factory(logger)
    def fit(self, X, y=None):
        X = np.copy(X)

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)

        return self

    @log_time_factory(logger)
    def transform(self, X, y=None):
        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return np.clip(X, [self.min_], [self.max_])


class OneHotEncoderStandard(OneHotEncoder):
    def __init__(self):
        super().__init__(sparse=False, handle_unknown="ignore")

    @log_time_factory(logger)
    def fit(self, X, y=None):
        return super().fit(X, y)

    @log_time_factory(logger)
    def transform(self, X):
        return super().transform(X)

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return super().inverse_transform(X)


class OutlierFiltering(LocalOutlierFactor, TransformerMixin):
    def __init__(self, *, max_frac: float = 0.01):
        super().__init__(novelty=True)

        self.max_frac = max_frac

    @log_time_factory(logger)
    def fit(self, X, y=None):
        return super().fit(X, y)

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        outliers = self.predict(X) == -1

        if np.mean(outliers) > self.max_frac:
            max_outliers = int(len(outliers) * self.max_frac)

            outliers[outliers] = np.random.permutation(
                [True] * max_outliers + [False] *
                (np.sum(outliers) - max_outliers)
            )

        return X[~outliers, :]

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return X


class ConditionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    @log_time_factory(logger)
    def fit(self, X, y=None):
        X = np.copy(X)

        self.less_condition_ = []

        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i == j:
                    continue

                if np.mean(X[:, i] <= X[:, j]) > self.threshold:
                    self.less_condition_.append((i, j))

        return self

    @log_time_factory(logger)
    def transform(self, X, y=None):
        X = np.copy(X)

        for i, j in self.less_condition_:
            swap = X[:, i] > X[:, j]

            if np.any(swap):
                temp = X[swap, i]
                X[swap, i] = X[swap, j]
                X[swap, j] = temp

        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return self.transform(X)


class QuantileTransformer_(QuantileTransformer):
    from sklearn.utils._testing import ignore_warnings

    @ignore_warnings(
        category=UserWarning if os.getenv(
            "YDATA_ENV", "PROD") == "PROD" else tuple()
    )
    def fit(self, X, y=None):
        return super().fit(X, y)


class DatetimeTimestamp(TransformerMixin):
    @log_time_factory(logger)
    def fit(self, X, y=None):
        return self  # Stateless operator

    @log_time_factory(logger)
    def transform(self, X):
        """Convert a datetime to a date a timestamp i.e. number of second
        elapsed since Jan 1970.

        Timestamp in second
        """
        return pd.to_datetime(X).view("int64") // 10**9

    @log_time_factory(logger)
    def inverse_transform(self, X):
        """Convert a datetime to a date a timestamp i.e. number of second
        elapsed since Jan 1970.

        Timestamp in second
        """
        return X.view("datetime64[s]")


# Gaussianization Pipeline
class BayesianGM(BaseEstimator, TransformerMixin):
    """Bayesian Gaussian Mixture according to CTGAN paper (for ctgan)"""

    def __init__(self, weight_treshold: float = 0.005, max_clusters: int = 10):
        self.weight_treshold = weight_treshold
        self.max_clusters = max_clusters

    @log_time_factory(logger)
    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y=None):
        """'For' loop to fit n models to n continuous columns."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._gm_models = []
        for column in X:
            gm = BayesianGaussianMixture(
                n_components=10,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=0.001,
                n_init=1,
            )
            gm.fit(np.array(X[column]).reshape(-1, 1))
            valid_component_indicator = gm.weights_ > self.weight_treshold
            num_components = valid_component_indicator.sum()
            self._gm_models.append(
                {
                    "model": gm,
                    "used_comp_bool": valid_component_indicator,
                    "num_comp": num_components,
                }
            )

        return self

    @log_time_factory(logger)
    def transform(self, X, y=None):
        """Transform function that output the new vector [scalar,onehot] where
        the scalar is the normalized value and the onehot is a vector
        representing the mode."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        list_to_output = []
        for i, model_info in enumerate(self._gm_models):
            gm = model_info["model"]
            valid_component_indicator = model_info["used_comp_bool"]
            num_components = model_info["num_comp"]

            column_data = X.iloc[:, [i]]
            column_data = column_data.values

            # Calculating the meanss and stds
            means = gm.means_.reshape((1, self.max_clusters))
            stds = np.sqrt(gm.covariances_).reshape((1, self.max_clusters))
            normalized_values = ((column_data - means) / (4 * stds))[
                :, valid_component_indicator
            ]
            component_probs = gm.predict_proba(column_data)[
                :, valid_component_indicator
            ]

            selected_component = np.zeros(len(column_data), dtype="int")
            for i in range(len(column_data)):
                component_porb_t = component_probs[i] + 1e-6
                component_porb_t = component_porb_t / component_porb_t.sum()
                selected_component[i] = np.random.choice(
                    np.arange(num_components), p=component_porb_t
                )

            # Here is selecting the scalar
            selected_normalized_value = normalized_values[
                np.arange(len(column_data)), selected_component
            ].reshape([-1, 1])
            selected_normalized_value = np.clip(
                selected_normalized_value, -0.99, 0.99)

            # Here is creating the one_hot
            selected_component_onehot = np.zeros_like(component_probs)
            selected_component_onehot[
                np.arange(len(column_data)), selected_component
            ] = 1
            list_to_output += [selected_normalized_value,
                               selected_component_onehot]

        transformed = np.concatenate(list_to_output, axis=1).astype(float)
        return transformed

    @log_time_factory(logger)
    def inverse_transform(self, X):
        """From processed to raw."""
        # Calculating the positions of the columns
        st = 0
        restored_columns = []
        for model_info in self._gm_models:
            # the plus 1 is to get the dimension output
            gm = model_info["model"]
            dim = model_info["num_comp"] + 1
            column_data = X[:, st: st + dim]

            # Taking the scalar and the one_hot
            selected_normalized_value = column_data[:, 0]
            selected_component_probs = column_data[:, 1:]

            selected_normalized_value = np.clip(
                selected_normalized_value, -1, 1)
            component_probs = np.ones(
                (len(column_data), self.max_clusters)) * -100
            component_probs[:, model_info["used_comp_bool"]
                            ] = selected_component_probs

            means = gm.means_.reshape([-1])
            stds = np.sqrt(gm.covariances_).reshape([-1])
            selected_component = np.argmax(component_probs, axis=1)

            std_t = stds[selected_component]
            mean_t = means[selected_component]
            column = selected_normalized_value * 4 * std_t + mean_t

            restored_columns.append(column)
            st += dim

        recovered_data = np.column_stack(restored_columns)
        return recovered_data


class SimpleCategoricalImputer(BaseEstimator, TransformerMixin):
    """Simple Categorical Imputer.

    It simply inputs the most frequent value found in the column.
    It uses the SimpleImputer of Scikitlearn. Here it's using the simple imputer for each column for this reason:
        using the strategy 'most_frequent' would work on the dataframe as a whole but it would takes too much time,
        in this way it goes faster.
    source: https://datascience.stackexchange.com/questions/66034/sklearn-simpleimputer-too-slow-for-categorical-data-represented-as-string-values
    """

    @log_time_factory(logger)
    def fit(self, X, y=None):
        self._imputers = []
        for col in X.columns:
            imp = SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value=X[col].mode()[0]
            )
            imp.fit(X[[col]])
            self._imputers.append(imp)
        return self

    @log_time_factory(logger)
    def transform(self, X):
        # The copy is to avoid the "SettingWithCopyWarning"
        X = X.copy()
        for col, imp in zip(X.columns, self._imputers):
            X.loc[:, [col]] = imp.transform(X.loc[:, [col]])
        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return X

# also used in timeseries pipeline


class DateTimeImputer(BaseEstimator, TransformerMixin):
    @log_time_factory(logger)
    def fit(self, X, y=None):
        return self

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        return pd.DataFrame(X).ffill().bfill().values

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return X


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    COLS_PER_FEATURE = {
        "timestamp": 1,
        "month_ohe": 12,
        "month_cyclic": 2,
        "day_cyclic": 2,
        "day_of_week_ohe": 7,
        "hour_cyclic": 2,
        "minute_cyclic": 2,
    }

    def __init__(self, *, dt_features=None):
        assert dt_features is None or set(sum(dt_features, [])) < set(
            self.COLS_PER_FEATURE.keys()
        )

        self.dt_features = dt_features

        self.month_ohe = OneHotEncoder(
            categories=[list(range(1, 13))], sparse=False, handle_unknown="ignore"
        )
        self.dow_ohe = OneHotEncoder(
            categories=[list(range(7))], sparse=False, handle_unknown="ignore"
        )

    @staticmethod
    def _cyclic(x: np.ndarray, min_value: float, max_value: float):
        assert len(x.shape) == 1
        assert min_value < max_value

        x -= min_value
        x /= max_value - min_value + 1
        x *= 2 * np.pi

        return np.stack([np.sin(x), np.cos(x)], axis=1)

    @log_time_factory(logger)
    def fit(self, X, y=None):
        assert self.dt_features is None or len(self.dt_features) == X.shape[1]

        if self.dt_features is None:
            self.dt_features_ = [["timestamp"] for _ in range(X.shape[1])]
        else:
            self.dt_features_ = self.dt_features

        return self

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        result = []
        for i in range(X.shape[1]):
            dt = pd.to_datetime(pd.Series(X[:, i]))

            for feature in self.dt_features_[i]:
                if feature == "timestamp":
                    value = dt.astype(np.int64).values.reshape(-1, 1) // 10**9
                elif feature == "month_ohe":
                    value = self.month_ohe.fit_transform(
                        dt.dt.month.to_frame())
                elif feature == "month_cyclic":
                    value = self._cyclic(dt.dt.month, 1, 12)
                elif feature == "day_cyclic":
                    value = self._cyclic(dt.dt.day, 1, 31)
                elif feature == "day_of_week_ohe":
                    value = self.dow_ohe.fit_transform(
                        dt.dt.dayofweek.to_frame())
                elif feature == "hour_cyclic":
                    value = self._cyclic(dt.dt.hour, 0, 23)
                elif feature == "minute_cyclic":
                    value = self._cyclic(dt.dt.minute, 0, 59)
                else:
                    raise ValueError

                result.append(value)

        return np.concatenate(result, axis=1)

    @log_time_factory(logger)
    def inverse_transform(self, X):
        result = []
        idx = 0
        for features in self.dt_features_:
            result.append(pd.to_datetime(
                X[:, idx] * 10**9, errors="coerce").values)
            idx += sum([self.COLS_PER_FEATURE[k] for k in features])

        return np.stack(result, axis=1)

    def get_feature_names(self, input_features=None):
        if input_features is None:
            raise NotImplementedError("TODO: implement")

        assert len(input_features) == len(self.dt_features_)

        result = []
        for name, features in zip(input_features, self.dt_features_):
            for feature in features:
                result.append(f"{name}_{feature}")

        return result


class DecimalPlaces(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._num_to_round = []
        for i in range(len(X.T)):
            decimal = X[:, i][1]
            decimal = abs(Decimal(str(decimal)).as_tuple().exponent)
            self._num_to_round.append(decimal)
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        for i, num in enumerate(self._num_to_round):
            X[:, i] = np.round(X[:, i], num)
        return X


# FIXME needs to extract skewness ratio method from the class
class Gaussianization(BaseOperator):
    name: str = "Gaussianization"

    # Threshold on the ratio of columns with high skeweness
    skewed_threshold: float = 0.75

    def __init__(self, metadata: Metadata):
        super().__init__()

        config = RBIGConfig(
            uniformizer="hist",
            rotation="PCA",
            max_iter=100,
            zero_tolerance=20,
            bins=100,  # auto is not optimal
            alpha=1e-10,
            eps=1e-10,
            max_layers=1_000,
            domain_hint=None,
        )
        self.transformer = RBIG(config)
        self.column_encoder = {}
        self.input_types = {}
        self.output_types = {}
        self.mask_MV = {}
        self._transformed_columns = []
        # Threshold on value frequency to consider a high skeweness as extreme
        self.extreme_skewness_threshold = 0.5
        self.min_interval = {}
        self.counts = {}
        self.extreme_frequent = {}
        self.metadata = metadata
        self._skewed_ratio = 0.

    @staticmethod
    def get_min_interval(dist):
        vals = sorted(dist.index.to_list())
        diff = [vals[i+1] - vals[i]
                for i, e in enumerate(vals[:-1])]
        return min(diff) if len(diff) else 0

    @staticmethod
    def skewness_ratio(metadata):
        transformed_columns = Gaussianization.transformed_columns(
            {k: v for k, v in metadata.columns.items()})
        skewed_columns = [w.column for w in metadata.warnings['skewness']]
        skewed_nums = [
            c for c in transformed_columns if transformed_columns if c in skewed_columns]
        if len(transformed_columns) > 0:
            return len(skewed_nums) / len(transformed_columns)
        return 0.

    @staticmethod
    def transformed_columns(dtypes):
        return [k for k, v in dtypes.items() if v.vartype == VariableType.FLOAT and v.datatype == DataType.NUMERICAL]

    @log_time_factory(logger)
    def fit(self, X, input_dtypes):
        self._input_dtypes = deepcopy(input_dtypes)
        self._output_dtypes = deepcopy(input_dtypes)

        self._transformed_columns = self.transformed_columns(
            self._input_dtypes)
        self._skewed_ratio = self.skewness_ratio(self.metadata)

        if self._skewed_ratio > Gaussianization.skewed_threshold:
            self._transformed_columns = []
            return self

        if len(self._transformed_columns) > 0:
            X_ = X[self._transformed_columns]
            for k in self._transformed_columns:
                m = X_[k].isna()
                if m.sum() > 0:
                    self.mask_MV[k] = m

                counts = X_[k].value_counts()
                dist = counts / X_.shape[0]
                extreme_frequent = dist[dist >
                                        self.extreme_skewness_threshold].index.to_list()
                if extreme_frequent:
                    self.extreme_frequent[k] = extreme_frequent
                    self.min_interval[k] = self.get_min_interval(dist)
                    X_[k] = X_[k].apply(lambda x: x + randint(0, counts[extreme_frequent[0]] - 1) * (
                        self.min_interval[k] / counts[extreme_frequent[0]]) if x in extreme_frequent else x)

            for k, mask in self.mask_MV.items():
                median = X_[k].median()
                X_[k] = X_[k].fillna(0.0 if np.isnan(median) else median)
                X[f"{k}|NaN"] = mask.astype(bool)
                self._output_dtypes[f"{k}|NaN"] = Column(
                    name=f"{k}|NaN",
                    datatype=DataType.CATEGORICAL,
                    vartype=VariableType.BOOL,
                )
            self.transformer.fit(nparray(X_))
        return self

    @log_time_factory(logger)
    def fit_transform(self, X, input_dtypes):
        self.fit(X, input_dtypes)
        return self.transform(X)

    @log_time_factory(logger)
    def transform(self, X):
        if len(self._transformed_columns) > 0:
            X_ = X[self._transformed_columns]
            for k, mask in self.mask_MV.items():
                mean = X_[k].mean()
                X_[k] = X_[k].fillna(mean if mean is not np.nan else 0.0)
                X[f"{k}|NaN"] = mask.astype(bool)
                self._output_dtypes[f"{k}|NaN"] = Column(
                    name=f"{k}|NaN",
                    datatype=DataType.CATEGORICAL,
                    vartype=VariableType.BOOL,
                )
            X_ = self.transformer.transform(X_.to_numpy())
            X[self._transformed_columns] = X_
        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        if len(self._transformed_columns) > 0:
            columns = X.columns
            X_ = X[self._transformed_columns].astype(float)
            X_ = self.transformer.inverse_transform(X_.to_numpy())
            X[self._transformed_columns] = X_
            for k, mask in self.mask_MV.items():
                X[f"{k}|NaN"] = X[f"{k}|NaN"].apply(
                    lambda x: {"True": True, "False": False}.get(x, x)
                )
                X.loc[X[f"{k}|NaN"].astype(bool), k] = None
            X.columns = columns
            for k, extreme_frequent in self.extreme_frequent.items():
                X[k] = X[k].apply(lambda x: next(
                    (c for c in extreme_frequent if x > c and x < c + self.min_interval[k]), x))
        return X


# Timeseries Pipeline
class CategoricalEncoder(OrdinalEncoder):
    @staticmethod
    def _get_missing_value(x):
        try:
            x[~pd.isna(x)].astype(int)
            return -1
        except ValueError:
            return "missing_value"

    @log_time_factory(logger)
    def fit(self, X, y=None):
        super().fit(X, y)

        return self

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        for i in range(len(self.categories_)):
            mask = np.isin(X[:, i], self.categories_[i])

            if ~np.all(mask):
                X[~mask, i] = self.categories_[i][0]

        return super().transform(X)

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return super().inverse_transform(X)


class CategoricalImputer(BaseEstimator, TransformerMixin):
    @staticmethod
    def _is_int(x):
        try:
            x[~pd.isna(x)].astype(int)
        except ValueError:
            return False
        else:
            return True

    @log_time_factory(logger)
    def fit(self, X, y=None):
        X = np.copy(X)

        self.int_cols_ = np.array([self._is_int(X[:, i])
                                  for i in range(X.shape[1])])

        return self

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        if self.int_cols_.any():
            X[:, self.int_cols_] = (
                SimpleImputer(strategy="constant", fill_value=-1)
                .fit_transform(X[:, self.int_cols_])
                .astype(int)
            )
        if (~self.int_cols_).any():
            X[:, ~self.int_cols_] = (
                SimpleImputer(strategy="constant", fill_value="missing_value")
                .fit_transform(X[:, ~self.int_cols_])
                .astype(str)
            )

        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        X[:, self.int_cols_] = X[:, self.int_cols_].astype(int)

        X = X.astype(np.object)
        X[(X == -1) | (X == "missing_value")] = np.nan

        return X


class IdentityTransformer(BaseEstimator, TransformerMixin):
    @log_time_factory(logger)
    def fit(self, X, y=None):
        return self

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return X


class IntegerTransformer(BaseEstimator, TransformerMixin):
    @log_time_factory(logger)
    def fit(self, X, y=None):
        return self

    @log_time_factory(logger)
    def transform(self, X):
        X = np.copy(X)

        return X

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return np.around(X).astype(int)


class NumericImputer(IterativeImputer):
    def __init__(self):
        verbose = 2 if logger.getEffectiveLevel() == logging.DEBUG else 0

        super().__init__(n_nearest_features=50, verbose=verbose)

    @log_time_factory(logger)
    def fit(self, X, y=None):
        return super().fit(X, y)

    @log_time_factory(logger)
    def transform(self, X):
        return super().transform(X)

    @log_time_factory(logger)
    def inverse_transform(self, X):
        return X


# Cart Hierarchical Pipeline
class CatEncoder(BaseOperator):
    name: str = "CatEncoder"

    def __init__(self, metadata: Metadata, vartypes=None, anonymize_config=None, col_per_encoder: int = 10):
        super().__init__()
        self.col_per_encoder: int = col_per_encoder
        self.categorical_columns: list = []
        self.encoders: list[OrdinalEncoder] = []
        self.encoders_cols: list = []
        self.value_counts: dict = metadata.summary["value_counts"]
        self.n_encoders: int = 0
        self.vartypes = (
            vartypes
            if vartypes is not None
            else [VariableType.STR, VariableType.BOOL, VariableType.FLOAT,  VariableType.DATE,  VariableType.DATETIME]
        )
        self.bool_cols_with_mv = [k for k, v in metadata.summary['missings'].items(
        ) if v > 0 and metadata.columns.get(k).vartype == VariableType.BOOL]
        if anonymize_config is None:
            anonymize_config = {}
        self.anonymize_cols = AnonymizerEngine.get_anonymized_columns(
            anonymize_config)

    def _categorize(self, data: ddDataFrame, categorical_columns: list) -> ddDataFrame:
        # We might be missing value_counts for high cardinality categories or datetime for instance
        # This is possible because we try to save time during the metadata computation
        missing_value_counts = [
            k for k in categorical_columns if k not in self.value_counts or k in self.anonymize_cols]

        if missing_value_counts:
            logger.debug(
                "[SYNTHESIZER] - Compute missing value counts for categorical columns: {}".format(','.join(missing_value_counts)))
            no_null = {col: drop_null(
                data[col], is_str=self._input_dtypes[col].vartype == VariableType.STR) for col in missing_value_counts}
            add_value_counts = {col: no_null[col].value_counts()
                                for col in missing_value_counts}
            self.value_counts.update(compute(add_value_counts)[0])
        return data

    @log_time_factory(logger)
    def fit(self, X: Dataset, input_dtypes: dict[str, Column]) -> "CatEncoder":
        return self.fit_or_fit_transform(X, input_dtypes, fit_transform=False)

    def fit_or_fit_transform(self, X: Dataset, input_dtypes: dict[str, Column], fit_transform: bool = False) -> Union[Dataset, "CatEncoder"]:
        self._input_dtypes = deepcopy(input_dtypes)
        self._output_dtypes = deepcopy(input_dtypes)
        self.categorical_columns = [
            k for k, v in input_dtypes.items() if v.datatype in CATEGORICAL_DTYPES and v.vartype in self.vartypes
        ]

        for k in self.categorical_columns:
            self._output_dtypes[k] = Column(
                k, DataType.CATEGORICAL, VariableType.INT
            )

        data = X.to_dask()
        self._categorize(data, self.categorical_columns)

        self.cols = X.columns
        self.n_encoders = len(self.cols) // self.col_per_encoder + \
            (1 if len(self.cols) % self.col_per_encoder > 0 else 0)

        self.cols_per_encoder = [self.cols[i * self.col_per_encoder:(
            i + 1) * self.col_per_encoder] for i in range(self.n_encoders)]
        self.cat_cols_per_encoder = [
            [c for c in cols if c in self.categorical_columns] for cols in self.cols_per_encoder]

        encoders = [OrdinalEncoder(self.cat_cols_per_encoder[i])
                    for i in range(self.n_encoders)]
        slices = [data[self.cols_per_encoder[i]]
                  for i in range(self.n_encoders)]
        slices_categorized = [d.astype(
            {k: pd.api.types.CategoricalDtype(list(
                self.value_counts[k].index)) for k in self.cols_per_encoder[i] if k in self.categorical_columns}
        ) for i, d in enumerate(slices)]

        if fit_transform:
            encoded_parts = [encoders[i].fit_transform(
                slices_categorized[i]) for i in range(self.n_encoders)]
            self.encoders = encoders
            if encoded_parts:
                X._data = concat(encoded_parts, axis='columns')
            return X

        self.encoders = [encoders[i].fit(
            slices_categorized[i]) for i in range(self.n_encoders)]

        return self

    @log_time_factory(logger)
    def transform(self, X: Dataset) -> Dataset:
        data = X.to_dask()
        cols_per_encoder = [[c for c in cols if c in X.columns]
                            for cols in self.cols_per_encoder]
        cat_cols_per_encoder = [
            [c for c in cols if c in self.categorical_columns] for cols in cols_per_encoder]

        slices = [data[cols_per_encoder[i]] for i in range(self.n_encoders)]
        slices_categorized = [d.astype(
            {k: pd.api.types.CategoricalDtype(list(
                self.value_counts[k].index)) for k in cols_per_encoder[i] if k in self.categorical_columns}
        ) for i, d in enumerate(slices)]

        encoded_parts = [self._get_adjusted_encoder(
            i, data[cols_per_encoder[i]], cat_cols_per_encoder[i]).transform(
            slices_categorized[i]) for i in range(self.n_encoders)]
        encoded_parts = [e for e in encoded_parts if e.shape[1] > 0]
        if encoded_parts:
            X._data = concat(encoded_parts, axis='columns')
        return X

    @log_time_factory(logger)
    def fit_transform(self, X: Dataset, input_dtypes: Dict[str, Column]) -> Dataset:
        logger.debug(
            "[SYNTHESIZER] - Executing the categorical encoder preprocessing step.")
        return self.fit_or_fit_transform(X, input_dtypes, fit_transform=True)

    @log_time_factory(logger)
    def inverse_transform(self, X: ddDataFrame) -> ddDataFrame:
        cols = [col for col in self.categorical_columns if col in X.columns]

        cols_per_encoder = [[c for c in cols if c in X.columns]
                            for cols in self.cols_per_encoder]
        cat_cols_per_encoder = [
            [c for c in cols if c in self.categorical_columns] for cols in cols_per_encoder]

        # This is required only for Multitable!

        X = X.astype({k: int for k in cols})

        decoded_parts = [
            self._get_adjusted_encoder(
                i, X[cols_per_encoder[i]], cat_cols_per_encoder[i])
            .inverse_transform(X[cols_per_encoder[i]]) for i in range(self.n_encoders)]
        X = concat(decoded_parts, axis=1)

        # This is needed only because Dask cannot handle missing values in boolean columns
        bool_cols = [c for c in cols if self._input_dtypes[c].vartype ==
                     VariableType.BOOL and c in self.bool_cols_with_mv]
        if bool_cols:
            X[bool_cols] = X[bool_cols].replace({"True": 1.0, "False": 0.0})
        return X

    def _get_adjusted_encoder(self, i: int, data: ddDataFrame, categorical_columns: list):
        """Adjust the ith encoder to the categorical columns presents in the
        provided dataframe.

        The inverse transform accept any dataset, including some with
        less columns. To avoid the encoder to raise an error, we need to
        adjust its categorical columns.
        """
        oe = deepcopy(self.encoders[i])
        oe.columns_ = data.columns
        oe.categorical_columns_ = pd.Index(categorical_columns)
        oe.non_categorical_columns_ = pd.Index(
            [col for col in data.columns if col not in categorical_columns])
        return oe


class DateTimeToNumerical(BaseOperator):
    name: str = "DateTimeToNumerical"

    def __init__(self, sortbykey=None):
        super().__init__()
        self.input_types = {}
        self.output_types = {}
        self.transformed_columns = None
        self.sortbykey = sortbykey
        self.origin_dates = {}

    def fit(self, X: Dataset, input_dtypes: Dict[str, Column]):
        self._input_dtypes = input_dtypes
        self._output_dtypes = input_dtypes
        self.transformed_columns = [
            k for k, v in input_dtypes.items() if v.datatype == DataType.DATE
        ]
        return self

    def fit_transform(self, X: Dataset, input_dtypes: Dict[str, Column]):
        logger.debug(
            "[SYNTHESIZER] - Executing the datetime to numerical preprocessing step.")
        self.fit(X, input_dtypes)
        return self.transform(X)

    def transform(self, X: Dataset):
        transformed_columns = [
            c for c in self.transformed_columns if c in X.columns]
        for col in transformed_columns:
            X.to_dask()[col] = to_datetime(
                X.to_dask()[col]).view("int64") // 10**9
            self._output_dtypes[col] = Column(
                col, DataType.NUMERICAL, VariableType.INT)
        return X

    def inverse_transform(self, X: ddDataFrame) -> ddDataFrame:
        assign_kwargs = {}
        for col in self.transformed_columns:
            if col in X.columns:
                assign_kwargs[col] = X[col].astype("int64")[X[col].astype(
                    "int64") >= pd.Timestamp.min.timestamp()].view("datetime64[s]")
        X = X.assign(**assign_kwargs)
        return X


class AnonymizerEngine(BaseOperator):
    name: str = "AnonymizerEngine"

    def __init__(self, locale: str = None):
        super().__init__()
        self._locale = locale

    def fit(self, X: Dataset, input_dtypes: Dict[str, Column] = None, metadata: Metadata = None):

        metrics_logger.info(dataset=X,
                            datatype="tabular",
                            method='anonymizer')

        if input_dtypes:
            self._input_dtypes = deepcopy(input_dtypes)
            self._output_dtypes = deepcopy(input_dtypes)
        else:
            self._input_dtypes = deepcopy(metadata.columns)
            self._output_dtypes = deepcopy(metadata.columns)
        return self

    def fit_transform(self, X: Dataset, metadata: Metadata, config: dict | AnonymizerConfigurationBuilder, input_dtypes: Dict[str, Column] = None):
        logger.debug(
            "[SYNTHESIZER] - Executing the anonymization preprocessing step.")
        self.fit(X, input_dtypes=input_dtypes, metadata=metadata)
        return self.transform(X, metadata, config)

    def _update_dtypes(self, column: str, anonymizer_type):
        if anonymizer_type == AnonymizerType.INT:
            self._output_dtypes[column] = Column(
                column, DataType.CATEGORICAL, VariableType.INT
            )
            self._input_dtypes[column] = Column(
                column, DataType.CATEGORICAL, VariableType.INT
            )
        else:
            self._output_dtypes[column] = Column(
                column, DataType.CATEGORICAL, VariableType.STR
            )
            self._input_dtypes[column] = Column(
                column, DataType.CATEGORICAL, VariableType.STR
            )

    def transform(
        self,
        X: Dataset,
        metadata: Metadata = None,
        config: dict | AnonymizerConfigurationBuilder | None = None
    ):
        if not config or metadata is None:
            return X
        input_columns = X.columns
        if isinstance(config, dict):
            config_ = self.process_config(config)
            builder = AnonymizerConfigurationBuilder(config_, self._locale)
            config_ = builder.get_config()
        else:
            config_ = config.get_config()

        for k, v in config_.items():
            col_to_anonymize = [
                c for c in v.cols if c in self._input_dtypes and c in input_columns]
            if col_to_anonymize:
                card = metadata[col_to_anonymize].summary["cardinality"]
                try:
                    X.to_dask()[col_to_anonymize] = v.get_anonymizer()(
                        X.to_dask()[col_to_anonymize], card, **v.params)

                except MaxIteration:
                    v.params["unique"] = False
                    X.to_dask()[col_to_anonymize] = v.get_anonymizer()(
                        X.to_dask()[col_to_anonymize], card, **v.params)
                for c in col_to_anonymize:
                    self._update_dtypes(
                        column=c, anonymizer_type=v.type)
        return X

    def inverse_transform(self, X: ddDataFrame) -> ddDataFrame:
        return X

    @staticmethod
    def process_config(config: dict) -> dict:
        cfg = {}
        for k, v in config.items():
            cfg_e = {
                'cols': [],
                'type': None
            }
            # Remap the original configuration
            if isinstance(v, dict):
                # If does not contain the proper fields, will raise error later
                cfg_e = deepcopy(v)
                if 'cols' not in cfg_e:
                    cfg_e['cols'] = [k]
            else:
                cfg_e['cols'] = [k]
                cfg_e['type'] = v

            cfg[k] = cfg_e
        return cfg

    @staticmethod
    def get_anonymized_columns(config: dict | AnonymizerConfigurationBuilder):
        cols = []
        if isinstance(config, AnonymizerConfigurationBuilder):
            for conf in config.get_config().values():
                cols += conf.cols
        else:
            raw_config = AnonymizerEngine.process_config(config)
            for _, anom_cfg in raw_config.items():
                cols += anom_cfg["cols"]
        return list(set(cols))


class NumericalClipping(BaseOperator):
    name: str = "Clipping"

    def __init__(self, metadata: Metadata):
        super().__init__()
        self.input_types = {}
        self.output_types = {}
        self.domain = metadata.summary['domains']
        self.og_dtypes = metadata.columns

    def fit(self, X: Dataset, input_dtypes: dict[str, Column]):
        self._input_dtypes = deepcopy(input_dtypes)
        self._output_dtypes = deepcopy(input_dtypes)
        return self

    def fit_transform(self, X: Dataset, input_dtypes: dict[str, Column]):
        logger.debug(
            "[SYNTHESIZER] - Executing the numerical clipping preprocessing step.")
        self.fit(X, input_dtypes)
        return self.transform(X)

    def transform(self, X: Dataset):
        return X

    def inverse_transform(self, X: ddDataFrame) -> ddDataFrame:
        logger.info('[SYNTHESIZER] - Numerical clipping')

        #Get the numerical cols
        num_cols = [
            k for k, v in self._output_dtypes.items()
            if v.datatype == DataType.NUMERICAL and self.og_dtypes[k].vartype not in [VariableType.DATE,
                                                                                      VariableType.DATETIME]
        ]

        if len(num_cols) > 0:
            cols = [k for k in num_cols if k in self.domain and k in X.columns]
            """
            to_num = {
                k: int if self._input_dtypes[k].vartype == VariableType.INT else float for k in cols}


            X = X.astype(to_num)
            """

            X_ = {k: X[k].clip(self.domain[k]['min'], self.domain[k]['max'])
                  for k in cols}

            X = X.assign(**X_)
        return X


class FloatFormatter(BaseOperator):
    name: str = "Float Formatter"

    def __init__(self):
        super().__init__()
        self.input_types = {}
        self.output_types = {}
        self.decimals = {}
        self.ctx = Context()
        self.ctx.prec = 20

    def fit(self, X: Dataset, input_dtypes: dict[str, Column]):

        def count_decimals(x):
            return len(str(x).split(".")[1]) if '.' in str(x) else 0

        self._input_dtypes = deepcopy(input_dtypes)
        self._output_dtypes = deepcopy(input_dtypes)

        data = X.to_dask()
        numerical_cols = data.select_dtypes(include=['float'])
        p = numerical_cols.get_partition(0) #making this more efficient by computing only over 1 partition

        self.decimals = p.applymap(count_decimals).max().compute().to_dict()

        return self

    def fit_transform(self, X: Dataset, input_dtypes: dict[str, Column]):
        logger.debug(
            "[SYNTHESIZER] - Executing the float formatter preprocessing step.")
        self.fit(X, input_dtypes)
        return self.transform(X)

    def transform(self, X: Dataset):
        return X

    def inverse_transform(self, X: ddDataFrame) -> ddDataFrame:
        cols = list(self.decimals.keys())
        cols = [k for k in cols if k in X.columns]

        config = {k: self.decimals[k] for k in cols}
        return X.round(config)
