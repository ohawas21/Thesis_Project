from _typeshed import Incomplete
from pandas import DataFrame as pdDataframe
from ydata.preprocessing.utils import BaseTransformer

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
    regex: Incomplete
    id_cols: Incomplete
    def __init__(self) -> None: ...
    REGEX_DEFAULT: str
    def fit(self, X: pdDataframe, y: Incomplete | None = None, regex_expression: dict = None, columns: list | str = None):
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
    def transform(self, X: pdDataframe, y: Incomplete | None = None) -> pdDataframe:
        """Removes the ID column types from the synthesis process.

        X: pd.DataFrame with the data to be transformed
        """
    def inverse_transform(self, X: pdDataframe) -> pdDataframe:
        """Add new ID columns to the provided X dataset.

        Generates the new random identifiers based on the attribute
        regex_expression values.
        """
