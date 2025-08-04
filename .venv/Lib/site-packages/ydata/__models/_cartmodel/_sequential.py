"""
Hierarchical cart dataflow code definition for sequential data (time-series)
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Dict

import numpy as np
from dill import dump as pdump
from dill import load as pload
from numpy import nan, zeros
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries

from ydata.__models._cartmodel._common import (intialize_smoothing_strategy, validate_datatypes,
                                              visitor_seq_to_predictor_mat)
from ydata.__models._cartmodel._preprocess import SeqPreprocessor
from ydata.__models._cartmodel.maps import (ENABLED_DATATYPES, METHOD_TO_TYPE_TO_FUNCTION, METHODS_MAP, CART_FLAVOR,
                                           Smoothing)
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.synthesizers.base_synthesizer import BaseSynthesizer
from ydata.synthesizers.exceptions import NoInputDataTypesWarning
from ydata.utils.acceleration_device import Device
from ydata.utils.data_types import VariableType
from ydata.utils.random import RandomSeed

methods = METHODS_MAP[CART_FLAVOR.SEQ]
methods_map = METHOD_TO_TYPE_TO_FUNCTION[CART_FLAVOR.SEQ]


# #@typechecked
class SeqCartHierarchical(BaseSynthesizer):
    __name__ = "SeqCartHierarchical"
    DEVICE = Device.CPU
    __INDEX: str = "__index"

    def __init__(
        self,
        proper: bool = False,
        smoothing: str = "NA",
        default_method: str = "cart",
        random_state: RandomSeed = None,
        regression_order=5,
    ):
        """Sequential Hierarchical Cart initialization.

        Args:
            proper (bool): True is proper synthesization, False otherwise.d
            smoothing (str): Smoothing strategy. No smoothing applied by default.
            default_method (str): Default method for column synthesization.
            random_state (Optional[int]): Random state of the synthesizer.
            regression_order (int): Order for the auto-regressive columns extraction.
        """

        assert default_method in [
            "cart",
            "parametric",
            "xgboost",
        ], "Only 'cart', 'xgboost' and 'parametric' are valid default methods."

        # todo add init validations and remove arguments that belong to the fit
        self.proper = proper
        self.default_method = default_method
        self.random_state = random_state
        self.order = regression_order
        self.smoothing = Smoothing(smoothing)

        self.smoothing_strategy = None  # Initialized in `fit` based on self.smoothing
        self.visit_sequence = None
        self._predictor_matrix = None
        self.columns_info: dict | None = None
        self.max_n_samples = None

        self.col_to_method = None
        self.col_to_function = None

        self.origin_dates = {}

        self._processor = SeqPreprocessor()

    def fit(self, X: pdDataFrame, metadata: Metadata, dtypes: dict[str, Column] | None = None, extracted_cols: list[str] | None = None, bootstrapping_cols: list[str] | None = None) -> SeqCartHierarchical:
        """Fit the SeqCartHierarchical synthesizer models to the provided training data.

        Args:
            X (Dataset): Training data.
            metadata (Metadata): The meta info from the provided dataset
            extracted_cols (List[str]): List of columns which have been extracted.
            bootstrapping_cols (List[str]): List of columns to boostrap the process

        Returns:
            SeqCartHierarchical: Synthesizer instance
        """
        if dtypes is None:
            warnings.warn(
                "The argument dtypes was not specified. "
                "The synthesizer will use 'metadata.columns' as default value.",
                NoInputDataTypesWarning,
            )
            dtypes = metadata.columns

        if metadata.dataset_attrs:
            sortscolumns = metadata.dataset_attrs.entities + metadata.dataset_attrs.sortbykey
            X = X.sort_values(by=sortscolumns).reset_index(drop=True).copy()

        self.__validate_datatypes(X, metadata, dtypes)
        self.__set_properties(X, metadata, dtypes,
                              extracted_cols, bootstrapping_cols)

        X, self.columns_info = self._processor.preprocess(
            X=X, synth=self, columns_info=self.columns_info
        )

        # Extend visit sequence and predictor with the new columns extracted during preprocessing
        self.__set_visit_sequence()
        self.__set_predictor_matrix()

        # todo add the logic to add the defaults methods
        self.__set_methods(X)

        self.saved_methods = {}
        _predictor_matrix_columns = self._predictor_matrix.columns.to_numpy()

        for col in self.visit_sequence:
            # initialise the method
            # todo change the logic here due to the enums use
            # use self.method instead
            dtypes = {
                k: {"datatype": v.datatype, "vartype": v.vartype}
                for k, v in dict(self.columns_info).items()
            }
            dtype = dtypes[col]

            smoothing = self.smoothing_strategy.get(
                col, self.smoothing_strategy.get(
                    col.replace("_dummy", ""), False)
            )
            col_method = self.col_to_function[col](
                y_dtype=dtype,
                dtypes=dtypes,
                smoothing=smoothing,
                proper=self.proper,
                random_state=self.random_state,
                cluster=self.conditioning_values,
                order=self.order,
            )
            # fit the method
            col_predictors = _predictor_matrix_columns[
                self._predictor_matrix.loc[col].to_numpy() == 1
            ]
            col_predictors = [
                c for c in col_predictors if c not in self.skip_cols]
            col_method.fit(
                X=X[col_predictors].copy(),
                y=X[col].copy(),
                autoregressive=True if col in self.other_cols else False,
                dtypes=dtypes,
            )

            # save the method
            self.saved_methods[col] = col_method
        return self

    def sample(self, n_samples: int = 100, bootstrapping: pdDataFrame | None = None, random_state: RandomSeed = None) -> pdDataFrame:
        """Generate a sample of synthetic data.

        Args:
            n_samples (int): Sample size.
            bootstrapping (DataFrame, optional): Data for the bootstrapping columns

        Returns:
            pd.DataFrame: Synthetic data
        """
        dtypes = {
            k: {"datatype": v.datatype, "vartype": v.vartype}
            for k, v in dict(self.columns_info).items()
        }

        synth_df = pdDataFrame(
            data=zeros([n_samples, len(self.visit_sequence)]),
            columns=self.visit_sequence,
        )
        _predictor_matrix_columns = self._predictor_matrix.columns.to_numpy()

        if bootstrapping is not None:
            # some columns may have been duplicated in the segmentation step
            bootstrapping = bootstrapping.loc[:,
                                              ~bootstrapping.columns.duplicated()]
            bootstrapping, bootstrap_columns_info = self._processor.preprocess(
                X=bootstrapping, synth=self, columns_info={
                    k: v for k, v in self.columns_info.items() if k in bootstrapping.columns}
            )

        for col in self.visit_sequence:
            col_method = self.saved_methods[col]
            col_predictors = _predictor_matrix_columns[self._predictor_matrix.loc[col].to_numpy(
            ) == 1]
            col_predictors = [
                c for c in col_predictors if c not in self.skip_cols]

            if bootstrapping is not None and col in bootstrapping.columns:
                y_pred = bootstrapping[col]
            else:
                y_pred = col_method.predict(synth_df[col_predictors].copy(),
                                            autoregressive=True if col in self.other_cols else False,
                                            dtypes=dtypes,
                                            cluster=synth_df[self.conditioning_col].values
                                            if self.conditioning_col is not None
                                            else [1] * len(synth_df), random_state=random_state)[:n_samples]
            synth_df[col] = y_pred
            if synth_df[col].notna().any():
                if dtypes[col]["vartype"] == VariableType.INT:
                    col_flt = synth_df[col].astype(float)
                    fill_value = int((col_flt.max() + 1) // 1)
                    synth_df[col] = col_flt.fillna(fill_value).astype(
                        VariableType.INT.value
                    )
                    synth_df.loc[synth_df[col] == fill_value, col] = nan
                else:
                    if dtypes[col]["vartype"] != VariableType.DATETIME:
                        synth_df[col] = synth_df[col].astype(
                            dtypes[col]["vartype"].value
                        )

            if dtypes[col]["vartype"] == VariableType.INT:
                synth_df[col] = synth_df[col].astype(VariableType.INT.value)

        synth_df = self._processor.postprocess(
            X=synth_df, synth=self, columns_info=self.columns_info, random_state=random_state
        )

        synth_df = synth_df.sort_values(
            by=self.sortbykey_col).reset_index(drop=True)
        synth_df.drop(columns=SeqCartHierarchical.__INDEX,
                      inplace=True, errors="ignore")
        return synth_df

    def __set_properties(self, X: pdDataFrame, metadata: Metadata, dtypes: Dict, extracted_cols: list[str] | None = None, bootstrapping_cols: list[str] | None = None):
        """Initialize all properties prior to training the model.

        Args:
            X (Dataset): Training data.
            metadata (Metadata): The meta info from the provided dataset.
            extracted_cols (List[str]): List of columns which have been extracted.
        """
        self.__set_dataset_attrs(
            X, metadata, dtypes, extracted_cols=extracted_cols)
        self.__set_visit_sequence(bootstrapping_cols=bootstrapping_cols)
        self.__set_predictor_matrix()
        self.__set_methods(X)
        self.__set_smoothing()
        self.__set_skip_cols()

        self.max_n_samples = len(X)

    def __set_visit_sequence(self, bootstrapping_cols: list[str] | None = None):
        """Determine the optimal visit sequence."""
        _visit_sequence = self.sortbykey_col.copy()  # [SeqCartHierarchical.__INDEX]
        if self.conditioning_col is not None:
            _visit_sequence += [self.conditioning_col]
        _visit_sequence += self.entities

        if bootstrapping_cols:
            self.other_cols = [c for c in self.other_cols if c in bootstrapping_cols] + \
                [c for c in self.other_cols if c not in bootstrapping_cols]
        _visit_sequence += self.other_cols

        _visit_sequence = list(OrderedDict.fromkeys(_visit_sequence))
        assert len(_visit_sequence) == len(set(_visit_sequence)
                                           ), "The dataset attrs are not disjoint."

        for v in self.columns_info.keys():
            if v not in _visit_sequence:
                _visit_sequence.append(v)

        self.visit_sequence = _visit_sequence

    def __set_predictor_matrix(self):
        """Set the predictor matrix."""
        self._predictor_matrix = visitor_seq_to_predictor_mat(
            self.visit_sequence)

    def __set_smoothing(self):
        """Set the smoothing strategy."""
        self.smoothing_strategy = intialize_smoothing_strategy(
            self.smoothing, self.columns_info
        )

    def __get_sortby_dist(self, X):
        is_equal = {col: False for col in self.sortbykey_col}
        if len(self.entities) > 0:
            groups = X[self.sortbykey_col +
                       self.entities].groupby(self.entities)
            ngroups = len(groups)

            tmedian, tmean = {col: 0 for col in self.sortbykey_col}, \
                {col: 0 for col in self.sortbykey_col}

            for group in X[self.sortbykey_col + self.entities].groupby(self.entities):
                df = group[1]
                diff = df[self.sortbykey_col].sort_values(
                    self.sortbykey_col).diff().dropna()
                for col in self.sortbykey_col:
                    tmedian[col] += np.median(diff[col])
                    tmean[col] += diff[col].values.mean()

            for col in self.sortbykey_col:
                median = tmedian[col] / ngroups
                mean = tmean[col] / ngroups

                if mean - median < 1.0:
                    is_equal[col] = True
        else:
            diff = X[self.sortbykey_col].sort_values(
                self.sortbykey_col).diff().dropna()
            for col in self.sortbykey_col:
                median = np.median(diff[col])
                mean = diff[col].values.mean()
                if mean - median < 1.0:
                    is_equal[col] = True

        return is_equal

    def __set_methods(self, X):
        """Set the methods to be applied to synthesize each columns.

        Depending on each column DataType and VariableType, as well as
        the default method and metadata, we assign to each column a
        method.
        """
        col_to_method = {}

        # Get whether dates are equally distributed or not
        is_equal = self.__get_sortby_dist(X)

        for col in self.sortbykey_col:
            if is_equal[col]:
                col_to_method[col] = methods.EMPTY
            else:
                col_to_method[col] = methods.PERTURB

        for col in self.other_cols:
            col_to_method[col] = methods.CART
            if col in self.extracted_cols:
                col_to_method[col] = methods.EMPTY

        for col in self.columns_info.keys():
            if col not in col_to_method:  # Columns generated during preprocessing
                col_to_method[col] = methods.CART

        assert col_to_method[self.visit_sequence[0]] in [
            methods.SAMPLE,
            methods.PERTURB,
            methods.EMPTY,
        ], "The first column synthesis method should be of type Sample, Perturb or Empty. "

        self.col_to_method = col_to_method

        # The final function to call might depend on the datatype, so we re-map it accordingly
        self.col_to_function = {
            col: methods_map[col_method][self.columns_info[col].datatype]
            for col, col_method in self.col_to_method.items()
        }

        self.col_to_function = pdSeries(self.col_to_function)

    def __set_dataset_attrs(
        self,
        X: pdDataFrame,
        metadata: Metadata,
        dtypes: Dict,
        extracted_cols: list | None = None,
    ):
        """Set the dataset attributes.

        The dataset attributes help to synthesize the data while preserving privacy, utiliy and fidelity.

        Args:
            X (Dataset): Training data.
            metadata (Metadata): The meta info from the provided dataset
            extracted_cols (List[str]): List of columns which have been extracted.
        """
        if extracted_cols is None:
            self.extracted_cols = []
        else:
            self.extracted_cols = extracted_cols

        self.columns_info = dtypes
        self.sortbykey_col = None
        self.entities = None
        self.conditioning_col = None

        # TODO: Check metadata attributes format for timeseries
        entities = metadata.dataset_attrs.entities
        sortbykey_col = metadata.dataset_attrs.sortbykey

        self.entities = entities.copy()
        self.sortbykey_col = sortbykey_col
        self.conditioning_col = None
        selected_cols = self.entities + self.sortbykey_col + (
            [self.conditioning_col] if self.conditioning_col is not None else [])
        self.other_cols = list(self.extracted_cols)
        selected_cols += self.extracted_cols
        self.other_cols += [
            k for k in self.columns_info.keys() if k not in selected_cols
        ]

        if self.conditioning_col is not None:
            self.conditioning_values = X[self.conditioning_col].values
        else:
            self.conditioning_values = [1] * len(X)

    def __set_skip_cols(self):
        """Set the columns to skip during synthesization."""
        self.skip_cols = []

    @staticmethod
    def __validate_datatypes(X, metadata, dtypes):
        validate_datatypes(ENABLED_DATATYPES[CART_FLAVOR.SEQ], dtypes)
        '''
        for sbk in metadata.dataset_attrs.sortbykey:
            sortbykey_dtype = dtypes[sbk].datatype
            if sortbykey_dtype == DataType.DATE and not X[sbk].is_monotonic:
                # The problem occurs because sortbykey with DATE are generated by elapsed time distribution
                raise Exception("Dataset should be sorted by sortbykey when sortbykey has datatype DATE")
        '''

    def __getstate__(self):
        state = self.__dict__.copy()
        if "col_to_method" in state:
            del state["col_to_method"]  # Enable serialization
        return state

    def save(self, path: str):
        with open(path, "wb") as f:
            pdump(self, f)

    @staticmethod
    def load(path: str) -> SeqCartHierarchical:
        with open(path, "rb") as f:
            model = pload(f)
        return model
