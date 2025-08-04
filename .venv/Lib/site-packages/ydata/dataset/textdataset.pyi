import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from _typeshed import Incomplete
from typing import Any, Callable

class TextDataset:
    """
    A simplified text dataset class that handles data transformations and operations.

    Attributes:
        name (str): Name of the dataset
        _data (pa.Table): The actual data stored as a PyArrow table
        output (Dict[str, Any]): The processed output
    """
    name: Incomplete
    output: Incomplete
    def __init__(self, name: str, data: list[dict[str, Any]]) -> None:
        """
        Initialize the text dataset.

        Args:
            name (str): Name of the dataset
            data (List[Dict[str, Any]]): The data to process
        """
    @property
    def schema(self) -> dict[str, Any]: ...
    @property
    def nrows(self) -> int:
        """
        Get the number of rows in the dataset.

        Returns:
            int: Number of rows
        """
    @property
    def ncols(self) -> int:
        """
        Get the number of columns in the dataset.

        Returns:
            int: Number of columns
        """
    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the dataset.

        Returns:
            Tuple[int, int]: A tuple containing (number of rows, number of columns)
        """
    @property
    def columns(self) -> list[str]:
        """
        Get the list of column names in the dataset.

        Returns:
            List[str]: List of column names
        """
    def map(self, func: Callable[[dict[str, Any]], dict[str, Any]], lazy: bool = False, name: str | None = None) -> TextDataset:
        """
        Apply a function to each row of data.

        Args:
            func (Callable): Function to apply to each row
            lazy (bool): Whether to apply the function lazily
            name (Optional[str]): Name for the operation

        Returns:
            TextDataset: New dataset with transformed data
        """
    def zip(self, other: TextDataset, name: str | None = None) -> TextDataset:
        """
        Combine two TextDataset objects side by side using PyArrow operations.

        Args:
            other (TextDataset): The other dataset to combine with
            name (Optional[str]): Name for the combined dataset

        Returns:
            TextDataset: New dataset with combined columns

        Raises:
            ValueError: If the datasets have different numbers of rows
        """
    def select_columns(self, columns: list[str], name: str | None = None) -> TextDataset:
        """
        Select specific columns from the data.

        Args:
            columns (List[str]): Columns to select
            name (Optional[str]): Name for the operation

        Returns:
            TextDataset: New dataset with selected columns
        """
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the data to a pandas DataFrame.

        Returns:
            pd.DataFrame: The data as a DataFrame
        """
    def to_pyarrow(self) -> pa.Table:
        """
        Convert the data to a PyArrow table.

        Returns:
            pa.Table: The data as a PyArrow table
        """
    def to_dask(self, n_partitions: int = 1) -> dd.DataFrame:
        """
        Convert the data to a Dask DataFrame.

        Returns:
            dd.DataFrame: The data as a Dask DataFrame
        """
