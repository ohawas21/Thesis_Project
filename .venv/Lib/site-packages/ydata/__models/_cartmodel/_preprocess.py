"""cartmodel missing values specific processing."""
import warnings

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas import Series as pdSeries
from sklearn.preprocessing import LabelEncoder

from ydata.metadata.column import Column
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType, VariableType
from ydata.utils.random import RandomSeed

warnings.filterwarnings("ignore")

NAN_KEY = "nan"
NUMTOCAT_KEY = "numtocat"


# @typechecked
class Preprocessor:
    def __init__(self):
        self.processing_dict = {NUMTOCAT_KEY: {}, NAN_KEY: {}}
        self.dt_cols = None
        self.cat_cols_encoder = {}  # Encoder for categorical in case of missing values
        self.cat_str_encoder = {}  # Simple label encoder for Categorical/String

    def preprocess_cat_str(self, X, dtypes):
        for k, v in dtypes.items():
            if v.datatype in CATEGORICAL_DTYPES and v.vartype in [
                VariableType.STR,
                VariableType.BOOL,
            ]:
                le = LabelEncoder()
                col = le.fit_transform(X[k])
                X[k] = pdSeries(col)
                self.cat_str_encoder[k] = le
        return X

    def postprocess_cat_str(self, X):
        for k, le in self.cat_str_encoder.items():
            col = le.inverse_transform(X[k].astype(int))
            X[k] = pdSeries(col)
        return X

    def preprocess(self, X, synth, columns_info):
        assert (
            synth.__name__ == "CartHierarchical"
        ), "Please provide a synthesizer model of type CartHierarchical."
        columns = [c for c in synth.visit_sequence if c in X.columns]
        for col in columns:
            col_nan_indices = X[col].isna()

            # Uncomment code when the contna logic is implemented
            # cont_nan_indices = {v: X[col] == v for v in synth.cont_na.get(col, [])}
            col_nan_series = [
                (np.nan, col_nan_indices)
            ]  # + list(cont_nan_indices.items())

            col_all_nan_indices = pd.DataFrame(
                {index: value[1] for index, value in enumerate(col_nan_series)}
            ).max(axis=1)
            col_not_nan_indices = np.invert(col_all_nan_indices)

            # Transform as type category for categorical variables?

            # transform numerical columns in numtocat to categorical
            if hasattr(synth, "numtocat") and col in synth.numtocat:
                self.processing_dict[NUMTOCAT_KEY][col] = {
                    "dtype": columns_info[col],
                    "categories": {},
                }

                # Dealing With Non-NaN Values
                not_nan_values = X.loc[col_not_nan_indices, col].copy()
                X.loc[col_not_nan_indices, col] = pd.cut(
                    X.loc[col_not_nan_indices, col],
                    synth.catgroups[col],
                    labels=range(synth.catgroups[col]),
                    include_lowest=True,
                )

                grouped = pd.DataFrame(
                    {"grouped": X.loc[col_not_nan_indices,
                                      col], "real": not_nan_values}
                ).groupby("grouped")
                self.processing_dict[NUMTOCAT_KEY][col]["categories"] = (
                    grouped["real"].apply(np.array).to_dict()
                )

                # Dealing with NaN
                for index, (_, bool_series) in enumerate(col_nan_series):
                    nan_cat = synth.catgroups[col] + index
                    self.processing_dict[NUMTOCAT_KEY][col]["categories"][
                        nan_cat
                    ] = X.loc[bool_series, col].to_numpy()
                    X.loc[bool_series, col] = nan_cat

                # todo validate how can this be achieved
                X[col] = X[col].astype("category")
                columns_info[col].datatype = DataType.CATEGORICAL

            else:
                # NaNs in category columns
                # need to process NaNs only as all other categories will be taken care automatically
                if columns_info[col].datatype in CATEGORICAL_DTYPES:
                    X[col] = X[col].astype("category")
                    if col_nan_indices.any():
                        # TODO beware of 'NaN_category' naming
                        col_nan_category = "NaN_category"
                        self.processing_dict[NAN_KEY][col] = {
                            "dtype": columns_info[col].datatype,
                            "nan_value": col_nan_category,
                        }

                        X[col].cat.add_categories(
                            col_nan_category, inplace=True)
                        X[col].fillna(col_nan_category, inplace=True)
                        X[col] = X[col].astype("str")

                        self.cat_cols_encoder[col] = LabelEncoder()
                        X[col] = self.cat_cols_encoder[col].fit_transform(
                            X[col])

                # NaNs in numerical columns
                elif columns_info[col].datatype == DataType.NUMERICAL:
                    if col_all_nan_indices.any():
                        # insert new column in df
                        # TODO beware of '_NaN' naming
                        col_nan_name = col + "_NaN"
                        X.insert(X.columns.get_loc(col), col_nan_name, 0)

                        self.processing_dict[NAN_KEY][col] = {
                            "col_nan_name": col_nan_name,
                            "dtype": columns_info[col],
                            "nan_flags": {},
                        }

                        for index, (cat, bool_series) in enumerate(col_nan_series):
                            cat_index = index + 1
                            self.processing_dict[NAN_KEY][col]["nan_flags"][
                                cat_index
                            ] = cat
                            X.loc[bool_series, col_nan_name] = cat_index
                        X.loc[col_all_nan_indices, col] = 0.0

                        X[col_nan_name] = X[col_nan_name].astype("int")
                        columns_info[col_nan_name] = Column(
                            name=col,
                            datatype=DataType.CATEGORICAL,
                            vartype=VariableType("int"),
                        )

        # CATEGORICAL / String needs to be encoded into int
        # X = self.preprocess_cat_str(X, columns_info)
        return X, columns_info

    def postprocess(self, X: pd.DataFrame, random_state: RandomSeed = None):
        # X = self.postprocess_cat_str(X)
        rng = default_rng(seed=random_state)
        for col, encoder in self.cat_cols_encoder.items():
            X[col] = encoder.inverse_transform(X[col].astype(int).tolist())

        for col, processing_numtocat_col_dict in self.processing_dict[
            NUMTOCAT_KEY
        ].items():
            X[col] = X[col].astype(object)
            col_synth_df = X[col].copy()

            for category, category_values in processing_numtocat_col_dict[
                "categories"
            ].items():
                category_indices = col_synth_df == category
                X.loc[category_indices, col] = rng.choice(
                    category_values, size=category_indices.sum(), replace=True
                )
                # cast dtype back to original (float for int column with NaNs)
                if (
                    X[col].isna().any()
                    and processing_numtocat_col_dict["dtype"] == "int"
                ):
                    X[col] = X[col].astype(float)
                else:
                    X[col] = X[col].astype(
                        processing_numtocat_col_dict["dtype"])

        for col, processing_nan_col_dict in self.processing_dict[NAN_KEY].items():
            # NaNs in category columns
            # need to postprocess NaNs only all other categories will be taken care automatically
            if processing_nan_col_dict["dtype"] in CATEGORICAL_DTYPES:
                col_nan_value = processing_nan_col_dict["nan_value"]
                X[col] = X[col].astype(object)
                X.loc[X[col] == col_nan_value, col] = None
                X[col] = X[col].astype("category")

            # NaNs in numerical columns
            elif processing_nan_col_dict["dtype"].datatype == DataType.NUMERICAL:
                NA_col = processing_nan_col_dict["col_nan_name"]
                if NA_col not in X.columns:
                    continue
                for nan_flag, nan_value in processing_nan_col_dict["nan_flags"].items():
                    nan_flag_indices = X[NA_col].astype(int) == nan_flag
                    X.loc[nan_flag_indices, col] = nan_value
                X.drop(columns=NA_col, inplace=True)

        return X


# @typechecked
class SeqPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.processing_dict = {NUMTOCAT_KEY: {}, NAN_KEY: {}}
        self.dt_cols = None
        self.cat_cols_encoder = {}  # Encoder for categorical in case of missing values

    def preprocess(self, X, synth, columns_info) -> (pd.DataFrame, dict):
        columns = [c for c in synth.visit_sequence if c in X.columns]
        for col in columns:
            col_nan_indices = X[col].isna()

            # Uncomment code when the contna logic is implemented
            # cont_nan_indices = {v: X[col] == v for v in synth.cont_na.get(col, [])}
            col_nan_series = [
                (np.nan, col_nan_indices)
            ]  # + list(cont_nan_indices.items())

            col_all_nan_indices = pd.DataFrame(
                {index: value[1] for index, value in enumerate(col_nan_series)}
            ).max(axis=1)
            col_not_nan_indices = np.invert(col_all_nan_indices)

            # Transform as type category for categorical variables?

            # transform numerical columns in numtocat to categorical
            if hasattr(synth, "numtocat") and col in synth.numtocat:
                self.processing_dict[NUMTOCAT_KEY][col] = {
                    "dtype": columns_info[col],
                    "categories": {},
                }

                # Dealing With Non-NaN Values
                not_nan_values = X.loc[col_not_nan_indices, col].copy()
                X.loc[col_not_nan_indices, col] = pd.cut(
                    X.loc[col_not_nan_indices, col],
                    synth.catgroups[col],
                    labels=range(synth.catgroups[col]),
                    include_lowest=True,
                )

                grouped = pd.DataFrame(
                    {"grouped": X.loc[col_not_nan_indices,
                                      col], "real": not_nan_values}
                ).groupby("grouped")
                self.processing_dict[NUMTOCAT_KEY][col]["categories"] = (
                    grouped["real"].apply(np.array).to_dict()
                )

                # Dealing with NaN
                for index, (_, bool_series) in enumerate(col_nan_series):
                    nan_cat = synth.catgroups[col] + index
                    self.processing_dict[NUMTOCAT_KEY][col]["categories"][
                        nan_cat
                    ] = X.loc[bool_series, col].to_numpy()
                    X.loc[bool_series, col] = nan_cat

                # todo validate how can this be achieved
                X[col] = X[col].astype("category")
                columns_info[col].datatype = DataType.CATEGORICAL

            else:
                # NaNs in category columns
                # need to process NaNs only as all other categories will be taken care automatically
                if columns_info[col].datatype in CATEGORICAL_DTYPES:
                    X[col] = X[col].astype("category")
                    if col_nan_indices.any():
                        # TODO beware of 'NaN_category' naming
                        col_nan_category = "NaN_category"
                        self.processing_dict[NAN_KEY][col] = {
                            "dtype": columns_info[col].datatype,
                            "nan_value": col_nan_category,
                        }

                        X[col].cat.add_categories(
                            col_nan_category, inplace=True)
                        X[col].fillna(col_nan_category, inplace=True)
                        X[col] = X[col].astype("str")

                        self.cat_cols_encoder[col] = LabelEncoder()
                        X[col] = self.cat_cols_encoder[col].fit_transform(
                            X[col])

                # NaNs in numerical columns
                elif columns_info[col].datatype == DataType.NUMERICAL:
                    if col_all_nan_indices.any():
                        # insert new column in df
                        # TODO beware of '_NaN' naming
                        col_nan_name = col + "_NaN"
                        X.insert(X.columns.get_loc(col), col_nan_name, 0)

                        self.processing_dict[NAN_KEY][col] = {
                            "col_nan_name": col_nan_name,
                            "dtype": columns_info[col],
                            "nan_flags": {},
                        }

                        for index, (cat, bool_series) in enumerate(col_nan_series):
                            cat_index = index + 1
                            self.processing_dict[NAN_KEY][col]["nan_flags"][
                                cat_index
                            ] = cat
                            X.loc[bool_series, col_nan_name] = cat_index
                        X.loc[col_all_nan_indices, col] = 0.0

                        X[col_nan_name] = X[col_nan_name].astype("int")
                        columns_info[col_nan_name] = Column(
                            name=col,
                            datatype=DataType.CATEGORICAL,
                            vartype=VariableType("int"),
                        )

        # CATEGORICAL / String needs to be encoded into int
        # X = self.preprocess_cat_str(X, columns_info)
        return X, columns_info

    def postprocess(self, X, synth, columns_info, random_state: RandomSeed = None):
        rng = default_rng(seed=random_state)
        # X = self.postprocess_cat_str(X)
        for col, encoder in self.cat_cols_encoder.items():
            X[col] = encoder.inverse_transform(X[col].astype(int).tolist())

        for col, processing_numtocat_col_dict in self.processing_dict[
            NUMTOCAT_KEY
        ].items():
            X[col] = X[col].astype(object)
            col_synth_df = X[col].copy()

            for category, category_values in processing_numtocat_col_dict[
                "categories"
            ].items():
                category_indices = col_synth_df == category
                X.loc[category_indices, col] = rng.choice(
                    category_values, size=category_indices.sum(), replace=True
                )
                # cast dtype back to original (float for int column with NaNs)
                if (
                    X[col].isna().any()
                    and processing_numtocat_col_dict["dtype"] == "int"
                ):
                    X[col] = X[col].astype(float)
                else:
                    X[col] = X[col].astype(
                        processing_numtocat_col_dict["dtype"])

        for col, processing_nan_col_dict in self.processing_dict[NAN_KEY].items():
            # NaNs in category columns
            # need to postprocess NaNs only all other categories will be taken care automatically
            if processing_nan_col_dict["dtype"] in CATEGORICAL_DTYPES:
                col_nan_value = processing_nan_col_dict["nan_value"]
                X[col] = X[col].astype(object)
                X.loc[X[col] == col_nan_value, col] = None
                X[col] = X[col].astype("category")

            # NaNs in numerical columns
            elif processing_nan_col_dict["dtype"].datatype == DataType.NUMERICAL:
                NA_col = processing_nan_col_dict["col_nan_name"]
                for nan_flag, nan_value in processing_nan_col_dict["nan_flags"].items():
                    nan_flag_indices = X[NA_col].astype(int) == nan_flag
                    X.loc[nan_flag_indices, col] = nan_value
                X.drop(columns=NA_col, inplace=True)
        return X
