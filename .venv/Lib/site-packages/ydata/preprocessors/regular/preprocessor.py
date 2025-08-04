import os
import warnings
from collections import namedtuple
from os import getenv
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ydata.metadata.metadata import Metadata
from ydata.preprocessors.activations import Activation
from ydata.preprocessors.base import Preprocessor
from ydata.preprocessors.preprocess_methods import (AnonymizerEngine, BayesianGM, CatEncoder, DateTimeImputer,
                                                    DateTimeToNumerical, DateTimeTransformer, DecimalPlaces,
                                                    FloatFormatter, Gaussianization, NumericalClipping, NumericImputer,
                                                    OneHotEncoderStandard, SimpleCategoricalImputer)
from ydata.synthesizers.logger import synthlogger_config
from ydata.utils.data_types import DataType
from ydata.utils.misc import log_time_factory

if os.getenv("YDATA_ENV", "PROD") == "PROD":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")

ActInfo = namedtuple("ActInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    ["column_name", "column_type", "output_info", "output_dimensions"],
)


class CartHierarchicalPreprocessor(Preprocessor):
    def __init__(
        self,
        sortbykey=None,
        anonymize_config: Optional[dict] = None,
        metadata: Metadata = None
    ):
        Preprocessor.__init__(
            self,
            steps=[
                ("anonymization", AnonymizerEngine()),
                ("datetime", DateTimeToNumerical(sortbykey=sortbykey)),
                ("encoder", CatEncoder(metadata=metadata,
                                       anonymize_config=anonymize_config)),
                ("clipping", NumericalClipping(metadata=metadata)),
                ("float_precision", FloatFormatter()),
            ],
        )


class CartHierarchicalSegmentPreprocessor(Preprocessor):
    def __init__(self, metadata: Metadata = None):
        Preprocessor.__init__(
            self, steps=[("gaussianization", Gaussianization(metadata=metadata))])


class DataProcessor(BaseEstimator, TransformerMixin):
    """Main class for the Data Preprocessing. It is a base version. It works
    like any other transformer in scikit learn with the methods fit, transform
    and inverse transform.

    Args:
        num_cols (list of strings):
            List of names of numerical columns
        cat_cols (list of strings):
            List of names of categorical columns
        dt_cols (list of strings):
            List of names of datetime columns
    """

    SUPPORTED_DTYPES = [DataType.NUMERICAL, DataType.STR,
                        DataType.CATEGORICAL, DataType.DATE]

    def __init__(
        self,
        *,
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        dt_cols: List[str] = None
    ):

        self.num_cols = [] if num_cols is None else num_cols
        self.cat_cols = [] if cat_cols is None else cat_cols
        self.dt_cols = [] if dt_cols is None else dt_cols

        self.num_pipeline = Pipeline(
            [
                ("imputer", NumericImputer()),
                ("decimals", DecimalPlaces()),
                ("scaler", MinMaxScaler()),
                ("gmm", BayesianGM()),
            ]
        )

        self.cat_pipeline = Pipeline(
            [
                ("imputer", SimpleCategoricalImputer()),
                ("encoder", OneHotEncoderStandard()),
            ]
        )

        self.dt_pipeline = Pipeline(
            [
                ("imputer", DateTimeImputer()),
                ("datetime", DateTimeTransformer()),
                ("scaler", MinMaxScaler()),
            ]
        )

    def __create_metadata_synth(self):
        col_transform_info = []
        # Numerical ls named tuple
        if self.num_cols:
            for col_name, gm_model in zip(
                self.num_cols, self.num_pipeline["gmm"]._gm_models
            ):
                col_transform_info.append(
                    ColumnTransformInfo(
                        col_name,
                        DataType.NUMERICAL,
                        [
                            ActInfo(1, Activation.TANH),
                            ActInfo(gm_model["num_comp"], Activation.SOFTMAX),
                        ],
                        gm_model["num_comp"],
                    )
                )
        # Categorical ls named tuple
        if self.cat_cols:
            for col_name, num_categories in zip(
                self.cat_cols, self.cat_pipeline["encoder"].categories_
            ):
                col_transform_info.append(
                    ColumnTransformInfo(
                        col_name,
                        DataType.CATEGORICAL,
                        [ActInfo(len(num_categories), Activation.SOFTMAX)],
                        len(num_categories),
                    )
                )

        # Temporary they just are being treated like numeric, so MinMax is applied.
        # So I just create the ActInfo with 1 activation TANh, output dimension 1 (output of MinMax).
        if self.dt_cols:
            for col_name in self.dt_cols:
                col_transform_info.append(
                    ColumnTransformInfo(
                        col_name, DataType.DATE, [
                            ActInfo(1, Activation.TANH)], 1
                    )
                )

        self.col_transform_info = col_transform_info

    @log_time_factory(logger)
    def fit(self, X, y=None):
        logger.info("Start data preprocessing.")
        self.col_order_ = [
            c for c in X.columns if c in self.num_cols + self.cat_cols + self.dt_cols
        ]
        self._types = X.dtypes

        self.num_pipeline.fit(X[self.num_cols]) if self.num_cols else np.zeros(
            [len(X), 0]
        )
        self.cat_pipeline.fit(X[self.cat_cols]) if self.cat_cols else np.zeros(
            [len(X), 0]
        )
        self.dt_pipeline.fit(
            X[self.dt_cols]) if self.dt_cols else np.zeros([len(X), 0])
        self.__create_metadata_synth()

        return self

    @log_time_factory(logger)
    def transform(self, X, y=None):
        num_data = (
            self.num_pipeline.transform(X[self.num_cols])
            if self.num_cols
            else np.zeros([len(X), 0])
        )
        cat_data = (
            self.cat_pipeline.transform(X[self.cat_cols])
            if self.cat_cols
            else np.zeros([len(X), 0])
        )
        dt_data = (
            self.dt_pipeline.transform(X[self.dt_cols])
            if self.dt_cols
            else np.zeros([len(X), 0])
        )

        transformed = np.concatenate([num_data, cat_data, dt_data], axis=1)

        self.num_col_idx_ = num_data.shape[1]
        self.cat_col_idx_ = self.num_col_idx_ + cat_data.shape[1]
        self.dt_col_idx_ = self.cat_col_idx_ + dt_data.shape[1]

        logger.info("End data preprocessing.")
        return transformed

    @log_time_factory(logger)
    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X)

    @log_time_factory(logger)
    def inverse_transform(self, X) -> pd.DataFrame:
        num_data, cat_data, dt_data, _ = np.split(
            X, [self.num_col_idx_, self.cat_col_idx_, self.dt_col_idx_], axis=1
        )

        num_data = (
            self.num_pipeline.inverse_transform(num_data)
            if self.num_cols
            else np.zeros([len(X), 0])
        )
        cat_data = (
            self.cat_pipeline.inverse_transform(cat_data)
            if self.cat_cols
            else np.zeros([len(X), 0])
        )
        dt_data = (
            self.dt_pipeline.inverse_transform(dt_data)
            if self.dt_cols
            else np.zeros([len(X), 0])
        )

        result = pd.concat(
            [
                pd.DataFrame(num_data, columns=self.num_cols),
                pd.DataFrame(cat_data, columns=self.cat_cols),
                pd.DataFrame(dt_data, columns=self.dt_cols),
            ],
            axis=1,
        )

        result = result.loc[:, self.col_order_]

        for col in result.columns:
            result[col] = result[col].astype(self._types[col])

        return result
