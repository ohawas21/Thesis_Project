"""
    Cart-hierarchical dataflow code definition
"""
from __future__ import annotations

import pickle
import warnings
from typing import Dict, Optional, Union

from numpy import zeros
from pandas import DataFrame, Series
from pandas.api.types import is_integer_dtype

from ydata.__models._cartmodel._common import (intialize_smoothing_strategy, validate_datatypes,
                                              visitor_seq_to_predictor_mat)
from ydata.__models._cartmodel._preprocess import Preprocessor
from ydata.__models._cartmodel.maps import (ENABLED_DATATYPES, METHOD_TO_TYPE_TO_FUNCTION, METHODS_MAP, NA_METHODS,
                                           CART_FLAVOR, Smoothing)
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.synthesizers.base_synthesizer import BaseSynthesizer
from ydata.synthesizers.exceptions import NoInputDataTypesWarning
from ydata.utils.acceleration_device import Device
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed

methods = METHODS_MAP[CART_FLAVOR.TAB]
methods_map = METHOD_TO_TYPE_TO_FUNCTION[CART_FLAVOR.TAB]


# #@typechecked
class CartHierarchical(BaseSynthesizer):
    __name__ = "CartHierarchical"
    FLAVOR = CART_FLAVOR.TAB
    DEVICE = Device.CPU

    def __init__(
        self,
        proper: bool = False,
        smoothing: str = "NA",
        default_method: str = "cart",
        random_state: RandomSeed = None,
    ):
        """Tabular cart initialization.

        Args:
            proper (bool): True is proper synthesization, False otherwise.d
            smoothing (str): Smoothing strategy. No smoothing applied by default.
            default_method (str): Default method for column synthesization.
            random_state: Random state of the synthesizer.
        """

        # todo add init validations and remove arguments that belong to the fit
        # Init all the parameters
        self.visit_sequence = None
        self.predictor_matrix = None
        self.columns_info = None
        self.proper = proper
        self.smoothing = Smoothing(smoothing)
        self.smoothing_strategy = None  # Initialized in `fit` based on self.smoothing

        assert default_method in [
            "cart",
            "parametric",
        ], "Only 'cart' and 'parametrics are valid default methods."
        self.default_method = default_method
        self.col_to_method = None
        self.col_to_function = None

        # todo implement in a second iteration (move to the fit as it depends on the input data)
        # self.numtocat = numtocat
        # self.catgroups = catgroups

        self.random_state = random_state

        self._processor = Preprocessor()

    def fit(self, X: DataFrame,
            metadata: Metadata,
            dtypes: Dict[str, Column] = None,
            method: Optional[Union[list,
                                   METHODS_MAP[CART_FLAVOR.TAB]]] = None,
            cont_na: dict | None = None,
            bootstrapping_cols: list[str] | None = None) -> CartHierarchical:
        """Fit the cart synthesizer models to the provided training data.

        Args:
            X (Dataset): Training data.
            metadata (Metadata): The meta info from the provided dataset.
            method (List[str]): List of methods to apply on each columns.
            cont_na: (Dict): Dictionary indicating the missing values replacement for continuous variables.
            bootstrapping_cols (List[str]): List of columns to boostrap the process (not used for regular at the moment)

        Returns:
            SeqCart: Synthesizer instance
        """
        # todo visit sequence logic :
        # todo add numtocat logic
        # todo add nan predictor_matrix logic
        # todo add catgroups logic
        if dtypes is None:
            warnings.warn(
                "The argument dtypes was not specified. "
                "The synthesizer will use 'metadata.columns' as default value.",
                NoInputDataTypesWarning,
            )
            dtypes = metadata.columns

        self.columns_info = dtypes
        self.__validate_datatypes(self.columns_info)

        # Setting the synth required properties and attributes
        self.__set_properties(
            method=method if method is not None else self.default_method,
            cont_na=cont_na,
        )

        # Process the existing missing data and update columns_info the the new nan_info columns
        X, self.columns_info = self._processor.preprocess(
            X=X, synth=self, columns_info=self.columns_info
        )

        # Extend visit sequence and predictor with the new columns extracted during preprocessing
        all_cols = list(self.columns_info.keys())
        if bootstrapping_cols is None:
            self.visit_sequence = all_cols
        else:
            # For now, the _NaN columns are not included in the conditional sampling logic.
            self.visit_sequence = [c for c in all_cols if c in bootstrapping_cols] + \
                [c for c in all_cols if c not in bootstrapping_cols]

        self.__set_predictor_matrix()

        # todo add the logic to add the defaults methods
        self.__set_methods(method=method)

        self.saved_methods = {}
        # train
        self._predictor_matrix_columns = self._predictor_matrix.columns.to_numpy()
        dtypes = {
            k: {"datatype": v.datatype, "vartype": v.vartype}
            for k, v in dict(self.columns_info).items()
        }
        for col in self.visit_sequence:
            # initialise the method
            # todo change the logic here due to the enums use
            # use self.method instead
            col_method = self.col_to_function[col](
                y_dtype=self.columns_info[col].datatype,
                dtypes=dtypes,
                smoothing=self.smoothing_strategy.get(col, False),
                proper=self.proper,
                random_state=self.random_state,
            )
            # fit the method
            col_predictors = self._predictor_matrix_columns[
                self._predictor_matrix.loc[col].to_numpy() == 1
            ]
            col_method.fit(X=X[col_predictors], y=X[col], dtypes=dtypes)
            # save the method
            self.saved_methods[col] = col_method
        return self

    def sample(self, n_samples: int = 100, bootstrapping: Optional[DataFrame] = None, random_state: RandomSeed = None) -> DataFrame:
        """Generate a sample of synthetic data.

        Args:
            n_samples (int): Sample size.
            bootstrapping (DataFrame, optional): Data for the bootstrapping columns

        Returns:
            pd.DataFrame: Synthetic data
        """
        # here add the sample logic for the synth
        synth_df = DataFrame(
            data=zeros([n_samples, len(self.visit_sequence)]),
            columns=self.visit_sequence,
        )
        dtypes = {
            k: {"datatype": v.datatype, "vartype": v.vartype}
            for k, v in dict(self.columns_info).items()
        }

        if bootstrapping is not None:
            bootstrapping, bootstrap_columns_info = self._processor.preprocess(
                X=bootstrapping, synth=self, columns_info={
                    k: v for k, v in self.columns_info.items() if k in bootstrapping.columns}
            )

        for col in self.visit_sequence:
            # todo add logger here
            if bootstrapping is not None and col in bootstrapping.columns:
                y_pred = bootstrapping[col]
            else:
                # Reload the method used to train the synth for the column
                method = self.saved_methods[col]
                # predict the column with the method
                col_predictors = self._predictor_matrix_columns[
                    self._predictor_matrix.loc[col].to_numpy() == 1
                ]
                y_pred = method.predict(
                    synth_df[col_predictors], dtypes=dtypes, random_state=random_state)

            if is_integer_dtype(y_pred.dtype) or dtypes[col]["datatype"] == DataType.CATEGORICAL:
                synth_df[col] = y_pred
            else:
                synth_df[col] = y_pred.astype('float')
        # post-process the missing values identified within the data
        self._processor.postprocess(X=synth_df, random_state=random_state)
        return synth_df

    def __set_properties(self, method: str, cont_na: dict | None = None):
        """Initialize all properties prior to training the model.

        Args:
            method (str): Default method to apply on each column to synthesize.
            cont_na (Dict): Dictionary indicating the missing values replacement for continuous variables.
        """
        self.__set_visit_sequence()
        self.__set_predictor_matrix()
        # todo add the logic to add the defaults methods
        self.__set_methods(method=method)
        self.__set_smoothing()
        self.__set_contna(cont_na=cont_na)

    def __set_visit_sequence(self):
        """Determine the optimal visit sequence."""
        if self.visit_sequence is None:
            # TODO: Optimize the visit sequence based on the DataType/VarType
            self.visit_sequence = [col for col in self.columns_info]

    def __set_predictor_matrix(self):
        """Set the predictor matrix."""
        self._predictor_matrix = visitor_seq_to_predictor_mat(
            self.visit_sequence)

    def __set_smoothing(self):
        """Set the smoothing strategy."""
        self.smoothing_strategy = intialize_smoothing_strategy(
            self.smoothing, self.columns_info
        )

    def __set_contna(self, cont_na: dict | None = None):
        """Set the continuous NA strategy.

        Args:
            cont_na (Dict): Dictionary indicating the missing values replacement for continuous variables.
        """
        if cont_na is not None:
            assert all(col in list(self.columns_info.keys()) for col in cont_na), (
                "The columns provided in the cont_na property do not exist in the provided dataset."
                "Please validate your inputs."
            )

            assert all(
                DataType(self.columns_info[col].datatype) == DataType.NUMERICAL
                for col in cont_na
            ), (
                "Only numerical variables can be considered for the continuous na property."
                "Please validate your input."
            )

            self.cont_na = {
                col: col_cont_na
                for col, col_cont_na in self.cont_na.items()
                if methods[col] in NA_METHODS
            }

    def __set_methods(self, method: str | None = None):
        """Set the methods to be applied to synthesize each columns.

        Depending on each column DataType and VariableType, as well as the default method and metadata,
        we assign to each column a method.

        Args:
            method (str): Default method to apply on each column to synthesize.
        """
        _method = method if method is not None else self.default_method
        _method = methods[_method.upper()]

        col_to_method = {col: _method for col in self.columns_info.keys()}
        col_to_method[self.visit_sequence[0]] = methods.SAMPLE
        self.col_to_method = col_to_method

        col_to_function = {
            col: methods_map[self.col_to_method[col]][val.datatype]
            for col, val in self.columns_info.items()
        }
        self.col_to_function = Series(col_to_function)

    @staticmethod
    def __validate_datatypes(columns):
        validate_datatypes(ENABLED_DATATYPES[CART_FLAVOR.TAB], columns)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "col_to_method" in state:
            del state["col_to_method"]  # Enable serialization
        return state

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> CartHierarchical:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
