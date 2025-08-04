import pandas as pd
from _typeshed import Incomplete
from enum import Enum
from json import JSONEncoder
from ydata.dataset import Dataset
from ydata.metadata import Metadata

DATA_TYPES: Incomplete

class JsonSerializer(JSONEncoder):
    def default(self, o): ...

class Activation(Enum):
    """Activivation auxiliar class."""
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    @staticmethod
    def valid_activation(activation):
        """
        Parameters
        ----------
        object Activation type object

        Returns boolean
        -------
        """

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
def infer_pandas_dtypes(df: pd.DataFrame):
    """Returns a pandas DataFrame with types infered"""
def are_columns_matching(X: Dataset, metadata: Metadata):
    """Return whether the columns between a Dataset object and a Metadata
    object are the same.

    Args:
        X (Dataset): dataset to compare.
        metadata (Metadata): metadata to compare.

    Yields:
        bool: True if columns are matching (without taking the order into account)
    """
