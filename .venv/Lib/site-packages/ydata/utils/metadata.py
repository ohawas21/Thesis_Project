"""Metadata classes.

Auxiliary classes to support both the synthesizers and data preprocessor
methods
"""
import json
from enum import Enum
from json import JSONEncoder

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype, is_string_dtype

from ydata.dataset import Dataset
from ydata.metadata import Metadata

DATA_TYPES = ["float", "int", "id", "category",
              "bool", "datetime", "timedelta", "json"]


class JsonSerializer(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.float):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, pd.Timestamp):
            return o.isoformat()
        elif isinstance(o, pd.Timedelta):
            return o.isoformat()
        else:
            return super().default(o)


class Activation(Enum):
    """Activivation auxiliar class."""

    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"

    @staticmethod
    def valid_activation(activation):
        """
        Parameters
        ----------
        object Activation type object

        Returns boolean
        -------
        """
        result = False
        if activation in [Activation.SIGMOID, Activation.TANH, Activation.SOFTMAX]:
            result = True
        return result


def get_types(df: pd.DataFrame, int_as_cat_threshold: int = 20) -> dict:
    """Returns a dictionary of {feature: schema type} for a given pandas
    dataframe.

    The possible schema types (bool,category,datetime,float,id,int,json) depend on the original data type (bool, datetime, float, int, string).

    Possible types (data type -> schema type):
        float: float
        int: id, category, int
        bool: bool
        string: datetime, json, id, category
        datetime: datetime, timedelta
    """

    # coerce types
    df = df.convert_dtypes()

    result = {}
    for c in df.columns:
        # parse float
        if is_float_dtype(df[c].values):
            result[c] = "float"

        # parse integer
        elif is_integer_dtype(df[c].values):
            if df[c].nunique() == len(df):
                result[c] = "id"
            elif (
                df[c].nunique() < int_as_cat_threshold
                or c.lower().startswith("id")
                or c.lower().endswith("id")
            ):
                result[c] = "category"
            else:
                result[c] = "int"

        # parse bool
        elif is_bool_dtype(df[c].values):
            result[c] = "bool"

        # parse string dtypes -- datetime/timedelta/json/id/category
        elif is_string_dtype(df[c].values):
            try:
                # parse datetime dtype
                pd.to_datetime(df[c])
                result[c] = "datetime"
            except BaseException:
                try:
                    # XXX: disable timedelta auto-parsing due to parsing strings with dashes (e.g. "10-20")
                    raise ValueError
                    # parse timedelta dtype
                    pd.to_timedelta(df[c].fillna("nat"))
                    result[c] = "timedelta"
                except BaseException:
                    try:
                        # parse JSON dtype
                        df[c].apply(json.loads)
                        result[c] = "json"
                    except BaseException:
                        # parse string dtype
                        if df[c].nunique() == len(df):
                            result[c] = "id"
                        else:
                            result[c] = "category"

        elif is_datetime64_any_dtype(df[c].values):
            try:
                # parse datetime dtype
                pd.to_datetime(df[c])
                result[c] = "datetime"
            except BaseException:
                # parse timedelta dtype
                pd.to_timedelta(df[c].fillna("nat"))
                result[c] = "timedelta"

        else:
            raise ValueError(f"Unknown type for column: {c}")

    return result


def infer_pandas_dtypes(df: pd.DataFrame):
    "Returns a pandas DataFrame with types infered"

    def _get_pandas_dtype(col):
        """
        Infer datatype of a pandas column, process only if the column dtype is object.
        input:   col: a pandas Series representing a df column.

        Example:
            >>> df.apply(_get_pandas_dtype, axis=0)
        """
        if col.dtype == "object":
            try:
                col_new = pd.to_numeric(col.dropna().unique())
                return col_new.dtype
            except BaseException:
                try:
                    col_new = pd.to_datetime(
                        col.dropna().unique(), infer_datetime_format=True
                    )
                    return col_new.dtype
                except BaseException:
                    try:
                        col_new = pd.to_timedelta(col.dropna().unique())
                        return col_new.dtype
                    except BaseException:
                        return "object"
        else:
            return col.dtype

    dtypes = df.apply(_get_pandas_dtype, axis=0)
    return df.astype(dtypes.to_dict(), errors="ignore")


def are_columns_matching(X: Dataset, metadata: Metadata):
    """Return whether the columns between a Dataset object and a Metadata
    object are the same.

    Args:
        X (Dataset): dataset to compare.
        metadata (Metadata): metadata to compare.

    Yields:
        bool: True if columns are matching (without taking the order into account)
    """
    dataset_columns = X.columns
    metadata_columns = metadata.columns.keys()
    return (
        len(metadata_columns) == len(dataset_columns)
        and len(list(set(dataset_columns) - set(metadata_columns))) == 0
    )
