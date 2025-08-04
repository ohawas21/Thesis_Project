"""
    Simplified TextDataset class for handling data transformations and operations.
"""
from typing import List, Dict, Any, Callable, Optional, Tuple
from copy import deepcopy

import pandas as pd
import pyarrow as pa
import dask.dataframe as dd

class TextDataset:
    """
    A simplified text dataset class that handles data transformations and operations.

    Attributes:
        name (str): Name of the dataset
        _data (pa.Table): The actual data stored as a PyArrow table
        output (Dict[str, Any]): The processed output
    """
    def __init__(
        self,
        name: str,
        data: List[Dict[str, Any]],
    ):
        """
        Initialize the text dataset.

        Args:
            name (str): Name of the dataset
            data (List[Dict[str, Any]]): The data to process
        """
        self.name = name

        if not isinstance(data, pa.Table):
            self._data = pa.Table.from_pylist(data) if data else pa.Table.from_pylist([])
        else:
            self._data = data
        self.output = {}
        #self._schema = self._infer_schema()

    @property
    def schema(self) -> Dict[str, Any]:
        return self._data.schema

    @property
    def nrows(self) -> int:
        """
        Get the number of rows in the dataset.

        Returns:
            int: Number of rows
        """
        return len(self._data)

    @property
    def ncols(self) -> int:
        """
        Get the number of columns in the dataset.

        Returns:
            int: Number of columns
        """
        if not self._data:
            return 0
        return len(self._data.columns)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the dataset.

        Returns:
            Tuple[int, int]: A tuple containing (number of rows, number of columns)
        """
        return (self.nrows, self.ncols)

    @property
    def columns(self) -> List[str]:
        """
        Get the list of column names in the dataset.

        Returns:
            List[str]: List of column names
        """
        return self._data.column_names

    def map(
        self,
        func: Callable[[Dict[str, Any]], Dict[str, Any]],
        lazy: bool = False,
        name: Optional[str] = None,
    ) -> 'TextDataset':
        """
        Apply a function to each row of data.

        Args:
            func (Callable): Function to apply to each row
            lazy (bool): Whether to apply the function lazily
            name (Optional[str]): Name for the operation

        Returns:
            TextDataset: New dataset with transformed data
        """
        if lazy:
            # In lazy mode, we just store the function and return
            self._map_func = func
            self._map_name = name
            return self

        # Convert to pandas for easier row-wise operations
        df = self._data.to_pandas()
        transformed_data = [func(row.to_dict()) for _, row in df.iterrows()]
        return TextDataset(name=name or f"{self.name}_mapped", data=transformed_data)

    def zip(self, other: 'TextDataset', name: Optional[str] = None) -> 'TextDataset':
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
        if self.nrows != other.nrows:
            raise ValueError("Datasets must have the same number of rows to zip")

        if not isinstance(other, TextDataset):
            raise ValueError("Zip is only valid between two TextDatasets' instances.")

        # Combine the PyArrow tables side by side
        combined_table = pa.concat_tables([self._data, other._data], promote=True)

        return TextDataset(
            name=name or f"{self.name}_zipped",
            data=combined_table
        )

    def select_columns(self, columns: List[str], name: Optional[str] = None) -> 'TextDataset':
        """
        Select specific columns from the data.

        Args:
            columns (List[str]): Columns to select
            name (Optional[str]): Name for the operation

        Returns:
            TextDataset: New dataset with selected columns
        """
        if not self._data:
            return self

        # Convert to DataFrame for easier column selection
        df = deepcopy(self._data.select(columns))

        return TextDataset(
            name=name or f"{self.name}_selected",
            data=df
        )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the data to a pandas DataFrame.

        Returns:
            pd.DataFrame: The data as a DataFrame
        """
        return self._data.to_pandas()

    def to_pyarrow(self) -> pa.Table:
        """
        Convert the data to a PyArrow table.

        Returns:
            pa.Table: The data as a PyArrow table
        """
        return self._data

    def to_dask(self, n_partitions: int=1) -> dd.DataFrame:
        """
        Convert the data to a Dask DataFrame.

        Returns:
            dd.DataFrame: The data as a Dask DataFrame
        """
        return dd.from_pandas(self.to_pandas(), npartitions=n_partitions)

    def _build_repr(self) -> dict:
        dataset = {
            "Shape": (self.nrows, self.ncols),
            "Schema": pd.DataFrame(
                [
                    {"Column": col, "Variable type": type}
                    for col, type in zip(self.schema.names, self.schema.types)
                ]
            ),
        }
        return dataset
