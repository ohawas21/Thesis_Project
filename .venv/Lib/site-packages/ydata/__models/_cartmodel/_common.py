from typing import List

import pandas as pd
from numpy import zeros

from ydata.__models._cartmodel.maps import Smoothing
from ydata.utils.data_types import DataType


def visitor_seq_to_predictor_mat(visit_sequence: List[str]):
    """Generate a predictor matrix based on a visitor sequence.

    Args:
        visit_sequence (List[str]): visitor sequence used to construct the predictor matrix

    Returns:
        pd.DataFrame: Predictor matrix
    """
    predictor_matrix = zeros(
        [len(visit_sequence), len(visit_sequence)], dtype=int)
    predictor_matrix = pd.DataFrame(
        predictor_matrix, index=visit_sequence, columns=visit_sequence
    )
    visited_columns = []
    for col in visit_sequence:
        predictor_matrix.loc[col, visited_columns] = 1
        visited_columns.append(col)
    return predictor_matrix


def intialize_smoothing_strategy(smoothing_method: Smoothing, columns: List[str]):
    """Define the smoothing strategy per column depending on a smoothing
    method.

    Args:
        smoothing_method (Smoothing): Smoothing strategy
        columns (List[str]): List of columns on which Smoothing needs to be applied

    Returns:
        Dict[str, bool]: Dictionary indicating if a column needs to be smoothed or not
    """
    strategy = {}
    if smoothing_method == Smoothing.NA:
        strategy = {col: False for col in columns}
    elif smoothing_method == Smoothing.DENSITY:
        strategy = {
            col: val.vartype
            for col, val in columns.items()
            if DataType(val.datatype) == DataType.NUMERICAL
        }
    else:
        # TODO: validate the smoothing dict flow
        assert all(
            (
                smoothing_method == Smoothing.DENSITY
                and DataType.NUMERICAL(columns[col].datatype) == DataType.NUMERICAL
            )
            or smoothing_method == Smoothing.NA
            for col, smoothing_method in strategy.items()
        )
        numerical_cols = [
            col for col, val in columns.items() if val.datatype == DataType.NUMERICAL
        ]
        strategy = {
            col: (
                strategy.get(
                    col, False) == Smoothing.DENSITY and col in numerical_cols
            )
            for col in list(columns.keys())
        }
    return strategy


def validate_datatypes(enabled_types, columns):
    for col, v in columns.items():
        assert v.datatype in enabled_types, (
            f"{col} of data type {v.datatype.value} is not supported. "
            "Supported types are: {}.".format(
                ", ".join(map(lambda x: x.value, enabled_types))
            )
        )
