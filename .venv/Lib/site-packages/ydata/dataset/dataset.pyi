import dask
from dask.dataframe import DataFrame as ddDataframe, Series as ddSeries
from dask.dataframe.core import Scalar as ddScalar
from dask.delayed import Delayed as Delayed
from numpy import ndarray as ndarray
from pandas import DataFrame as pdDataframe, Series as pdSeries
from typing import Literal
from ydata.dataset.engines import VALID_ENGINES
from ydata.dataset.schemas import DatasetSchema as Schema
from ydata.utils.data_types import VariableType

class Dataset:
    """Dataset class provides the interface to handle data within YData's package.

    Arguments:
        df (Union[pandas.DataFrame, dask.DataFrame]): The data to be manipulated.
        schema (Optional[Dict]): Mapping of column names to variable types.
        sample (float): Fraction of the data to be sampled as the Dataset
        index (Optional[str]): Name of the column to be used as index, if any. This is an optional input, specially recommended for TimeSeries data.
        divisions (Optional[list | tuple]): This property is utilized by Dask, the underlying engine of the Dataset object, to enhance performance during parallel computing. It can be leveraged to optimize data processing efficiency and scalability.

    Properties:
        columns (list[str]): list of column names that are part of the Dataset schema
        nrows (tuple): number of rows from the Dataset
        ncols (int): number of columns
        shape (tuple): tuple of (nrows, ncols)
        memory_usage (int): number of bytes consumed by the underlying dataframe
        nmissings (int): total number of missings in Dataset
        infered_dtypes_count (Dict[str, Dict]): infered data types per column
        infered_dtypes (Dict[str, str]): infered data type per column
        dtypes (Dict[str, str]): mapping of data type per column, either provided or inferred
        index (str): Returns the name of the index column

    Magic Methods:
        __len__: equal to nrows
        __contains__: checks whether a column is in a Dataset
        __getitem__: equal to select_columns
    """
    def __init__(self, df: VALID_ENGINES, schema: dict[str, Schema] | None = None, sample: float = 0.2, index: str | dask.dataframe.core.Index | None = None, divisions: list | tuple | None = None) -> None: ...
    def copy(self) -> Dataset:
        """Copy a Dataset instance.

        Returns:
            dataset (Dataset): A new Dataset instance with the scame schema and index.
        """
    @property
    def ncols(self) -> int:
        """
        Property that returns the number of columns
        Returns:
            ncols (int): Number of columns
        """
    @property
    def nrows(self) -> int:
        """
        Property that returns the number of rows
        Returns:
            nrows (int): number of rows

        """
    @property
    def columns(self) -> list[str | int]:
        """
        Property that returns a list of column names.
        Returns:
            columns (list[str]): A list with the Dataset column names.
        """
    @property
    def index(self) -> str | None:
        '''
        "A property that returns the name of the index column
        Returns:
            index_name (str): index columns name
        '''
    @property
    def loc(self):
        '''Label location based indexer for selection. This method is inherited
        from Dask original LocIndexer implementation.

        >>> df.loc["b"]
        >>> df.loc["b":"d"]
        '''
    @property
    def schema(self) -> dict:
        """
        Property that returns a dictionary of the schema of the dataset.
        The dictionary follows the following structure: {column_name: variable_type}

        Returns:
            schema (dict): A dictionary with the schema of the dataset.
        """
    @schema.setter
    def schema(self, new_value: dict[str, VariableType | str]) -> None:
        """
        Set a new schema for the `Dataset` object.

        This method updates the schema definition, mapping column names to their
        respective data types. The schema plays a crucial role in defining how data
        is processed, validated, and interpreted within the dataset.
        Args:
            new_value:
             A dictionary where keys represent column names, and values specify
              the corresponding data types. The data types can either be instances of
              `VariableType` or string representations of types.
        """
    def apply(self, function: callable, axis: int | str = 1, raw: bool = False, args: tuple | None = None, meta: dict | list[tuple] | tuple | Dataset | None | str = '__no_default__') -> Dataset:
        """Parallelized version of apply.

        Only supported on the rows axis.
        To guarantee results with expected format, output metadata should be provided with meta argument.
        Arguments:
            function (callable): Function to apply to each row
            axis (Union[int, str]): 1/'columns' apply function to each row.
                0/'index' apply function to each column is not supported.
            raw (bool): Passed function operates on Pandas Series objects (False), or numpy arrays (True)
            args (Optional[Tuple]): Positional arguments to pass to function in addition to the array/series
            meta (Optional[Union[Dict, List[Tuple], Tuple, Dataset]]): A dictionary, list of tuples, tuple or dataset
                that matches the dtypes and column names of the output. This is an optional argument since it only
                certifies that Dask will use the correct metadata instead of infering which may lead to unexpected
                results.
        Returns:
            df (Dataset): A dataset object output of function.
        """
    def shape(self, lazy_eval: bool = True, delayed: bool = False) -> tuple[int | Delayed | None, int]:
        """
        Returns dataset shape as a tuple (rows, columns).

        Supports lazy evaluation of nrows, ncols is unexpensive and
        returned directly

        Args:
            lazy_eval (bool): Returns the currently computed values for nrows and ncols properties. Defaults to True.
            delayed (bool): If True, compute delayed properties instead for nrows and ncols. This is recommended to optimize DASK's DAG flow, the underlying computational engine of the Dataset.
                            Defaults to False.
        Returns:
            shape (tuple): a tuple with the shape the Dataset's shape
        """
    @property
    def memory_usage(self) -> ddSeries:
        """
        A property that returns the memory usage of the Dataset.
        Returns:
            memory_usage (Dask Series): Memory usage of the Dataset.
        """
    def missings(self, compute: bool = False) -> ddSeries | pdSeries:
        """Calculates the number of missing values in a Dataset.
        """
    @property
    def nmissings(self) -> int:
        """
        Get the total number of missing values in the `Dataset`.

        This property computes and returns the sum of missing values across all columns in the dataset,
        returning the total count as an integer.

        Returns:
            nmissings (int): The total number of missing values in the Dataset

        ### Notes:
            - If there are no missing values, the returned value will be `0`.
        """
    def infer_dtypes(self, schema: dict | None = None):
        """
        Infer and assign data types to dataset columns.

        This method determines the most representative variable type for each feature
        based on observed value distributions. If a `schema` is provided, it overrides
        the inferred types. Otherwise, the method analyzes the dataset and assigns
        data types accordingly.

        Args:
            schema (Optional[dict], optional): A dictionary where keys are column names and values are the manually assigned data types. If `None`, the method automatically infers types.
        """
    def select_dtypes(self, include: str | list | None = None, exclude: str | list | None = None) -> Dataset:
        """
       Return a subset of the dataset containing only specified data types.

        This method filters the dataset to include or exclude specific data types,
        allowing users to focus on relevant columns based on their types.
        Args:
            include (Optional[str | list]): Specifies the columns with the expected variable types to included in the resulting dataset.
            exclude (Optional[str | list]): Specifies the columns with the variable types that are expected to be excluded in the resulting dataset.

        Returns:
            dataset (Dataset): Subset of the dataset containing only columns with the specified variably types.
        """
    def astype(self, column: str, vartype: VariableType | str, format: str | None = None):
        '''
        Convert a column in the dataset to a specified data type.

        This method changes the data type of a specified column in the dataset, ensuring that
        the conversion follows the defined `VariableType` mappings. It also updates the dataset\'s
        internal schema to reflect the new type.

        Args:
            column (str): The name of the column in the dataset to be converted
            vartype (VariableType | str): The target data type for the column. Can be a `VariableType` instance or a string representation of a type (e.g., `"int"`, `"float"`, `"date"`).
                                          If `"date"` is specified, the conversion ensures the column is treated as a date and your able to define the date format following Python\'s formats.
            format: An optional format string used for date parsing when `vartype="date"` or `vartype="datetime"`. If `None`, default parsing rules are applied.
        '''
    def update_types(self, dtypes: list):
        '''
        Batch update data types for multiple columns in the dataset.

        This method allows updating the data types of multiple columns at once by providing a
        list of dictionaries, where each dictionary specifies a column name and the target variable type.

        Args:
            dtypes: A list of dictionaries, where each dictionary must contain:
                    - `"column"` (`str`): The name of the column to update.
                    - `"vartype"` (`VariableType | str`): The new data type for the column.
        '''
    def to_pandas(self) -> pdDataframe:
        """
        Converts the Dataset object to a pandas DataFrame
        Returns:
            dataset (pandas.DataFrame): Return the data from the Dataset objects as a pandas DataFrame
        """
    def to_numpy(self) -> ndarray:
        """
        Converts the Dataset object to a Numpy ndarray
        Returns:
            dataset (Numpy ndarray): Return the data from the Dataset objects as a Numpy ndarray
        """
    def to_dask(self) -> ddDataframe:
        """
        Converts the Dataset object to a DASK DataFrame
        Returns:
            dataset (DASK.DataFrame): Return the data from the Dataset objects as a DASK DataFrame
        """
    def value_counts(self, col: str, compute: bool = True) -> ddSeries | pdSeries:
        """
        Compute the frequency of unique values in a specified column.

        This method returns the count of occurrences for each unique value in the given column.
        By default, it computes the result eagerly, but it can also return a lazy Dask Series
        for efficient computation on large datasets.

        Args:
            col (str): The name of the column in the Dataset that we want to count the values.
            compute (bool, optional): Whether to compute or delay the count. Defaults to True.

        Returns:
            value_counts (Dask.Series, Pandas.Series): a Series with the value_counts.
        """
    def uniques(self, col: str, approx: bool = True, delayed: bool = False) -> int | ddScalar:
        """
        Compute the number of unique values in a column.

        This method calculates the distinct count of values in a given column, either **exactly**
        or using an **approximate** method for improved performance on large datasets. The
        result is stored for future reference when an exact count is computed.

        Args:
            col:  The column name for which to compute the number of unique values.
            approx (bool, optional):  If `True`, uses an **approximate** method to estimate the unique value count. If `False`, computes the **exact** count.
                                      Defaults to True.
            delayed (bool, optional): Whether to compute or delay the count. Defaults to False.

        Returns:
            nuniques (int, DASK Scalar): The number of unique values in the column.
        """
    def drop_columns(self, columns: str | list, inplace: bool = False) -> Dataset | None:
        """Drops specified columns from a Dataset.

        Args:
            columns (str or list): column labels to drop
            inplace (bool): if False, return a copy. Otherwise, drop inplace and return None.
        """
    def select_columns(self, columns: str | list, copy: bool = True) -> Dataset:
        """Returns a Dataset containing only a subset with the specified columns.
        If columns is a single feature, returns a Dataset with a single column.

        Args:
            columns (str or list): column labels to select
            copy (bool): if True, return a copy. Otherwise, select inplace and return self.
        """
    def query(self, query: str) -> Dataset:
        """
        Filter the dataset using a query expression.

        This method applies a **Pandas-style query** to filter the dataset based on
        the given condition. It returns a new `Dataset` containing only the rows that
        match the query.

        For more information check Dask's documentation on Dask dataframe query expression(https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.query.html).
        Args:
            query (str): The query expression to filter the dataset.

        Returns:
            dataset (Dataset): The dataset resulting from the provided query expression.
        """
    def sample(self, size: float | int, strategy: Literal['random', 'stratified'] = 'random', **strategy_params) -> Dataset:
        '''
        Generate a sampled subset of the dataset.

        This method returns a sample from the dataset using either **random sampling**
        or **stratified sampling**. The sample size can be defined as an absolute number
        of rows or as a fraction of the dataset.

        Args:
            size (Union[float, int]): size (number of rows) of the sampled subset
            strategy (str["random", "stratified"]): strategy used to generate a sampled subset

        Returns:
            dataset (Dataset): the sampled subset of the dataset.
        '''
    def reorder_columns(self, columns: list[str]) -> Dataset:
        """Defines the order of the underlying data based on provided 'columns'
        list of column names.

        Usage:
            >>> data.columns
            ['colA', 'colB', colC']
            >>> data.reorder_columns(['colB', 'colC']).columns
            ['colB', 'colC']
        """
    @property
    def divisions(self) -> tuple:
        """
        A property that returns the number of divisions set for the Dataset.
        Returns:
            divisions (tuple): the number of divisions set for the Dataset.
        """
    @property
    def known_divisions(self) -> bool: ...
    def sort_values(self, by: list[str], ignore_index: bool = True, inplace: bool = False) -> Dataset | None:
        """
        Sort the dataset by one or more columns.

        This method sorts the dataset based on the specified column(s), returning either a
        new sorted dataset or modifying the existing dataset in place.

        Args:
            by (List[str]): A list wit the name of the column(s) to sort.
            ignore_index (bool, optional): Whether to ignore index or not. Defaults to True.
            inplace (bool, optional): Whether to sort the dataset in-place. Defaults to False.

        Returns:
            dataset (Dataset): the sorted dataset in case inplace is set to False.
        """
    def sorted_index(self, by: list[str]) -> pdSeries:
        """
        Get the sorted index of the dataset based on specified columns.

        This method computes the order of the dataset when sorted by the given column(s).
        It returns a Pandas Series representing the index positions corresponding to the
        sorted dataset.
        Args:
            by (List[str]): A list wit the name of the column(s) to sort.

        Returns:
            index (pandas Series): A Pandas Series containing the sorted index positions.
        """
    def head(self, n: int = 5) -> pdDataframe:
        """
        Return the `n` first rows of a dataset.

        If the number of rows in the first partition is lower than `n`,
        Dask will not return the requested number of rows (see
        `dask.dataframe.core.head` and `dask.dataframe.core.safe_head`).
        To avoid this corner case, we retry using all partitions -1.

        Args:
            n (int): Number of rows that we want to select from the top of the dataset

        Returns:
            dataset (pandas DataFrame): A pandas DataFrame containing the first `n` rows.
        """
    def tail(self, n: int = 5) -> pdDataframe:
        """
        Return the `n` last rows of a dataset.

        If the number of rows in the first partition is lower than `n`,
        Dask will not return the requested number of rows (see
        `dask.dataframe.core.head` and `dask.dataframe.core.safe_head`).
        To avoid this corner case, we retry using all partitions -1.

        Args:
            n (int): Number of rows that we want to select from the bottom of the dataset

        Returns:
            dataset (pandas DataFrame): A pandas DataFrame containing the last `n` rows.
        """
    def __len__(self) -> int:
        """Implements utility to call len(Dataset) directly, returning the
        number of rows.

        Usage:
        >>> len(data)
        """
    def __contains__(self, key) -> bool:
        """True if key is in Dataset columns.

        Usage:
        >>> 'my_column' in data
        """
    def __getitem__(self, key) -> Dataset:
        """
        Usage:
        >>> data[ ['columnA', 'columnB'] ]
        """
