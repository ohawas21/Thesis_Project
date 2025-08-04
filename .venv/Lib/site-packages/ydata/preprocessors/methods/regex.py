"""File with the regex related method to be leveraged to improve the data
synthesis process."""
from typing import Union
from warnings import warn

from exrex import getone
from pandas import DataFrame as pdDataframe

from ydata.preprocessing.utils import BaseTransformer


# @typechecked
class RegexID(BaseTransformer):
    """
    RegexId method to generate fake ID and unique identifiers for a given dataset
    Attributes:
    ----------
    regex : dict
        Dict with the regex value assumed to generate each ID column records

    id_cols: list
        List with the id columns
    """

    def __init__(self):
        self.regex = None
        self.id_cols = None

    # default regex value
    REGEX_DEFAULT = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

    def _validate_data(
        self, X, y=None, regex_expression: dict = None, columns=None, **check_params
    ):
        # Here more validations related with the Metadata

        if regex_expression:
            if columns is not None:
                warn(
                    "Only the keys provided in the parameter regular_expression will be considered for Regex replacement."
                )
            check_cols = all(col in X.columns for col in list(
                regex_expression.keys()))
            assert (
                check_cols
            ), "The columns provided for the regular expressions must exist in Dataset columns."
            regex = regex_expression
            id_cols = list(regex_expression.keys())
        else:
            assert columns is not None, (
                "If no regular_expression is provided, "
                "then provide a column name or a list of columns to be considered."
            )
            if type(columns) == str:
                columns = list(columns)

            check_cols = all(col in X.columns for col in columns)
            assert check_cols, (
                "Please provide a valid list of columns. "
                "Some of the provided column names do not exist in the dataset provided."
            )
            # metadata must match X dataset columns
            id_cols = columns
            regex = {col: self.REGEX_DEFAULT for col in id_cols}

        return regex, id_cols

    def fit(
        self,
        X: pdDataframe,
        y=None,
        regex_expression: dict = None,
        columns: Union[list, str] = None,
    ):
        """Validates the input values. Identifies which columns are ID and need
        to be replaced.

        #TODO Inference of a valid regex without user input
        Parameters
        ----------
        regex_expression: dict with the column name and regex expression to be considered for the generation.
        columns: If regex_expression is not provided, the parameter columns is mandatory.
        If no regex expression if provided, the defaults regex is assumed.
        X: pd.Dataframe with the data to be fit
        """
        self.regex, self.id_cols = self._validate_data(
            X, columns=columns, regex_expression=regex_expression
        )
        return self

    def transform(self, X: pdDataframe, y=None) -> pdDataframe:
        """Removes the ID column types from the synthesis process.

        X: pd.DataFrame with the data to be transformed
        """
        return X[X.columns[~X.columns.isin(self.id_cols)]]

    def inverse_transform(self, X: pdDataframe) -> pdDataframe:
        """Add new ID columns to the provided X dataset.

        Generates the new random identifiers based on the attribute
        regex_expression values.
        """
        result = X.copy()
        limit = X.shape[0]
        for col, re in self.regex.items():
            try:
                result[col] = [getone(re, limit=50) for _ in range(limit)]
            except ValueError:
                raise Exception(
                    f"Please provide a valid regex expression for the {col} column."
                )
        return result
