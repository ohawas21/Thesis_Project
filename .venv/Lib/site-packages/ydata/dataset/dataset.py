"""Dataset definition file."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import dask
import dask.dataframe as dd
from dask.dataframe import DataFrame as ddDataframe
from dask.dataframe import Series as ddSeries
from dask.dataframe.core import Scalar as ddScalar
from dask.delayed import Delayed
from numpy import ndarray
from pandas import DataFrame as pdDataframe
from pandas import Series as pdSeries

from warnings import warn
from ydata.dataset.engines import VALID_ENGINES, to_dask, to_numpy, to_pandas
from ydata.dataset.schemas import DatasetSchema as Schema
from ydata.dataset.utils import humanize_dtypes
from ydata.utils.configuration import TextStyle
from ydata.utils.data_types import VariableType
from ydata.utils.exceptions import (DatasetException, DatasetAssertionError, VariableTypeRequestError,
                                    InvalidDatasetSample, InvalidDatasetSchema, InvalidDatasetTypeError)
from ydata.utils.type_inference import DEFAULT_TYPES, TypeConverter, default_inference


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

    def __init__(
        self,
        df: VALID_ENGINES,
        schema: dict[str, Schema] | None = None,
        sample: float = 0.2,
        index: str | dask.dataframe.core.Index | None = None,
        divisions: list | tuple | None = None,
    ):
        # Setting dataset, dataset type and index
        data = to_dask(df)
        self._index = index

        if all([val in ["object", "string"] for col, val in data.dtypes.items()]):
            warnings.warn(
                "All the input Variable Types were set as `string` or `object`. "
                "It is recommend to revise the VariableType settings for optimal results."
            )
        # This is the time-consuming step
        data = self.__set_index(
            data=data, index=self._index, divisions=divisions)
        if schema:
            try:
                # sanitize parameter
                self._schema = {
                    c: v if isinstance(v, Schema) else Schema(
                        column=c, vartype=VariableType(v))
                    for c, v in schema.items()
                }

                _schema = {
                    c: VariableType(v.vartype).value for c, v in self._schema.items()
                }

                assert all(
                    [col in data.columns for col in schema.keys()]
                ), "Not all the dataset columns are defined in the schema. Please validate your input."
                assert all(
                    [dtype in DEFAULT_TYPES for dtype in set(_schema.values())]
                ), f"Not all dtypes provided ({_schema}) are supported. Valid dtypes => ({DEFAULT_TYPES})"
                data = data[list(_schema.keys())]

                # THIS CONVERSIONS WILL BE DONE LATER BY astype FUNCTION
                # for k, v in _schema.items():
                #     if v in ["date", "time", "datetime"]:
                #         _schema[k] = "datetime64[ns]"
                # MOVED TO THE SAMPLE METHOD TO AVOID INVOKING COMPUTE.
                # elif v == "int" and df[k].isna().values.any() > 0:
                #     # Pandas/Dask cannot convert to int if missing value
                #     # See: https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
                #     schema[k] = "float"
                #     warnings.warn(
                #         "The column {} has missing data. "
                #         "Int is not a supported VariableType for columns with missing data."
                #     )

                date_cols = {}
                for k, v in _schema.copy().items():
                    col_info = self._schema[k]
                    if v in ['date', 'time', 'datetime'] and col_info.format is not None:
                        date_cols[k] = col_info
                        _schema.pop(k)
                    elif v in ['date', 'time', 'datetime'] and col_info.format is None:
                        date_cols[k] = col_info
                        _schema[k] = "datetime64[ns]"

                if len(date_cols.keys()) > 0:
                    for col, v in date_cols.items():
                        if v.format:
                            data[col] = dd.to_datetime(
                                data[col], format=v.format, errors="coerce")
                        else:
                            data[col] = dd.to_datetime(
                                data[col], errors="coerce")

            except ValueError as e:
                raise InvalidDatasetSchema(f"Please provide a valid schema. "
                                           f"The provided schema does not match the input df - {e}.")
        else:
            dtypes, _schema = humanize_dtypes(data.dtypes.to_dict())
            self._schema = {col: Schema(column=col, vartype=VariableType(v))
                            for col, v in dtypes.items()}

        self._data = data.astype(_schema)
        self._nrows = None
        self._dtypes_count = None
        self._missings = None
        self._n_uniques = {}
        self.__sample = sample

    def __set_sample(self, sample):
        self.__sample = sample
        return

    def copy(self) -> Dataset:
        """Copy a Dataset instance.

        Returns:
            dataset (Dataset): A new Dataset instance with the scame schema and index.
        """
        return Dataset(self._data, schema=self._schema, index=self.index)

    @staticmethod
    def __create_dask_dataframe_index(data: ddDataframe) -> ddDataframe:
        data = data.assign(idx=1)
        data.index = data["idx"].cumsum() - 1
        data = data.drop(columns=["idx"])
        data = data.set_index(data.index, sorted=True)
        data.divisions = data.compute_current_divisions()
        return data

    @staticmethod
    def __set_index(
        data: ddDataframe,
        index: str | dask.dataframe.core.Index | None = None,
        divisions: list | tuple | None = None,
    ) -> ddDataframe:
        """Asserts existence of the index column and sets it as new index."""
        if index is not None or data.index is not None:
            if isinstance(index, dask.dataframe.core.Index):
                data.index = index
            elif index:
                assert (
                    index in data.columns
                ), f"Provided index {index} does not exist in the dataframe columns."

            if divisions:
                data = data.repartition(divisions=divisions)
        else:
            data = Dataset.__create_dask_dataframe_index(data)

        return data

    @property
    def ncols(self) -> int:
        """
        Property that returns the number of columns
        Returns:
            ncols (int): Number of columns
        """
        return len(self.columns)

    @property
    def nrows(self) -> int:
        """
        Property that returns the number of rows
        Returns:
            nrows (int): number of rows

        """
        if self._nrows is None:
            self._nrows = len(self._data)
        return self._nrows

    @property
    def columns(self) -> list[str | int]:
        """
        Property that returns a list of column names.
        Returns:
            columns (list[str]): A list with the Dataset column names.
        """
        return list(self._data.columns)

    @property
    def index(self) -> Optional[str]:
        """
        "A property that returns the name of the index column
        Returns:
            index_name (str): index columns name
        """
        return self._index

    @property
    def loc(self):
        """Label location based indexer for selection. This method is inherited
        from Dask original LocIndexer implementation.

        >>> df.loc["b"]
        >>> df.loc["b":"d"]
        """
        from dask.dataframe.indexing import _LocIndexer

        return _LocIndexer(self._data)

    @property
    def schema(self) -> dict:
        """
        Property that returns a dictionary of the schema of the dataset.
        The dictionary follows the following structure: {column_name: variable_type}

        Returns:
            schema (dict): A dictionary with the schema of the dataset.
        """
        return {val.column: VariableType(val.vartype) for val in self._schema.values()}

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
        for col, val in new_value:
            old_schema = self._schema[col]
            self._schema[col] = Schema(
                column=col, vartype=VariableType(val), format=old_schema.format)

    def apply(
        self,
        function: callable,
        axis: Union[int, str] = 1,
        raw: bool = False,
        args: Optional[Tuple] = None,
        meta: Optional[Union[Dict, List[Tuple],
                             Tuple, Dataset]] | str = "__no_default__",
    ) -> Dataset:
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
        if axis not in [1, "columns"]:
            raise NotImplementedError(
                f"The axis argument {axis} is not supported. Please use 1 or columns."
            )
        if isinstance(meta, Dataset):
            meta = meta._data
        if not args:
            args = ()
        data = self._data.apply(function, axis=axis,
                                raw=raw, args=args, meta=meta)
        if isinstance(data, ddSeries):
            data = data.to_frame(name=data.name)
        return Dataset(data, index=self.index, divisions=self._data.divisions)

    def shape(
        self, lazy_eval=True, delayed=False
    ) -> tuple[int | Delayed | None, int]:
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
        if lazy_eval:
            return (self._nrows, self.ncols)
        else:
            if delayed:
                return (self.nrows, self.ncols)
            else:
                return (dask.compute(self.nrows)[0], self.ncols)

    @property
    def memory_usage(self) -> ddSeries:
        """
        A property that returns the memory usage of the Dataset.
        Returns:
            memory_usage (Dask Series): Memory usage of the Dataset.
        """
        return self._data.memory_usage()

    def missings(self, compute=False) -> Union[ddSeries, pdSeries]:
        """Calculates the number of missing values in a Dataset.
        """
        if self._missings is None:
            self._missings = self._data.isnull().sum()
        if compute and isinstance(self._missings, ddSeries):
            self._missings = self._missings.compute()
        return self._missings

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
        missings = self.missings(compute=True)
        return int(sum(missings.values))

    @property
    def __infered_dtypes_count(self) -> dict:
        """Calculates the inferred value types per column."""
        if self._dtypes_count is None:
            infered_dtypes = self._data.sample(frac=self.__sample).applymap(
                default_inference.infer
            )
            calc_dtypes = dd.compute(
                [infered_dtypes[col].value_counts()
                 for col in infered_dtypes.columns]
            )[0]
            self._dtypes_count = {col.name: col.to_dict()
                                  for col in calc_dtypes}
        return self._dtypes_count

    def infer_dtypes(self, schema: Optional[dict] = None):
        """
        Infer and assign data types to dataset columns.

        This method determines the most representative variable type for each feature
        based on observed value distributions. If a `schema` is provided, it overrides
        the inferred types. Otherwise, the method analyzes the dataset and assigns
        data types accordingly.

        Args:
            schema (Optional[dict], optional): A dictionary where keys are column names and values are the manually assigned data types. If `None`, the method automatically infers types.
        """
        # TODO : Infer only for columns with missing initialization
        if schema is None:
            schema = {}
            for (
                feat,
                counts,
            ) in self.__infered_dtypes_count.items():  # for each feature
                keys = list(counts.keys())
                if (
                    "bool" in counts and "int" in counts
                ):  # if themisinge are ints, all bool_ints should also be ints
                    counts["int"] += counts["bool"]
                    counts["bool"] = 0
                if (
                    "float" in counts and "int" in counts
                ):  # if there are floats, all ints should also be floats
                    counts["float"] += counts["int"]
                    counts["int"] = 0
                if "date" in counts and len(keys) > 1:
                    total_counts = sum(counts.values())
                    counts = dict.fromkeys(counts, 0)
                    counts["date"] = total_counts

                schema[feat] = max(counts, key=counts.get)

        # assign feature dtype if representativity is highest
        dtype_implementation = {k: TypeConverter.to_low(
            v) for (k, v) in schema.items()}
        self._data = self._data.astype(dtype_implementation)

        self._schema = {col: Schema(column=col, vartype=VariableType(v))
                        for col, v in schema.items()}

    def select_dtypes(
        self,
        include: str | list | None = None,
        exclude: str | list | None = None,
    ) -> Dataset:
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
        if include is None and exclude is None:
            raise InvalidDatasetTypeError(
                "Either 'include' or 'exclude' arguments must be provided."
            )
        elif include:
            return self.select_columns(self._select_dtypes_vars(include))
        elif exclude:
            return self.drop_columns(self._select_dtypes_vars(exclude))
        else:
            raise DatasetException(
                f"Could not determine how to select data types based on include ({include}) and exclude ({exclude}) arguments."
            )

    def _select_dtypes_vars(self, dtypes: Union[str, VariableType, list]) -> list:
        "Returns a list of variables based on their data type."
        # Accept either singular or list of values.
        _dtypes = dtypes if isinstance(dtypes, list) else [dtypes]
        _dtypes = [
            VariableType(dtype) if isinstance(dtype, str) else dtype
            for dtype in _dtypes
        ]
        return [k for (k, v) in self._schema.items() if VariableType(v.vartype) in _dtypes]

    def astype(self, column: str, vartype: Union[VariableType, str], format: Optional[str] = None):
        """
        Convert a column in the dataset to a specified data type.

        This method changes the data type of a specified column in the dataset, ensuring that
        the conversion follows the defined `VariableType` mappings. It also updates the dataset's
        internal schema to reflect the new type.

        Args:
            column (str): The name of the column in the dataset to be converted
            vartype (VariableType | str): The target data type for the column. Can be a `VariableType` instance or a string representation of a type (e.g., `"int"`, `"float"`, `"date"`).
                                          If `"date"` is specified, the conversion ensures the column is treated as a date and your able to define the date format following Python's formats.
            format: An optional format string used for date parsing when `vartype="date"` or `vartype="datetime"`. If `None`, default parsing rules are applied.
        """
        self._data = _astype(self._data, column, vartype=VariableType(
            vartype).value, format=format)

        self._schema[column] = Schema(
            column=column,
            vartype=VariableType(
                TypeConverter.from_low(self._data[column].dtype))
            if VariableType(vartype) != VariableType.DATE else VariableType.DATE
        )

    def update_types(self, dtypes: list):
        """
        Batch update data types for multiple columns in the dataset.

        This method allows updating the data types of multiple columns at once by providing a
        list of dictionaries, where each dictionary specifies a column name and the target variable type.

        Args:
            dtypes: A list of dictionaries, where each dictionary must contain:
                    - `"column"` (`str`): The name of the column to update.
                    - `"vartype"` (`VariableType | str`): The new data type for the column.
        """
        invalid_dtypes = [e for e in dtypes if any(
            m not in e for m in ['column', 'vartype'])]
        if len(invalid_dtypes):
            raise VariableTypeRequestError('Invalid dtype update request:\n {}\n All items must specify a `column` and `vartype`'.format(
                ', \n '.join(map(str, invalid_dtypes))))

        invalid_dtypes = [e['column']
                          for e in dtypes if e['column'] not in self._schema.keys()]
        if len(invalid_dtypes):
            raise VariableTypeRequestError(
                'Invalid dtype update request. The following columns do not exist: {}'.format(', '.join(invalid_dtypes)))

        for e in dtypes:
            self.astype(**e)

    def to_pandas(self) -> pdDataframe:
        """
        Converts the Dataset object to a pandas DataFrame
        Returns:
            dataset (pandas.DataFrame): Return the data from the Dataset objects as a pandas DataFrame
        """
        df = to_pandas(self._data)
        return df

    def to_numpy(self) -> ndarray:
        """
        Converts the Dataset object to a Numpy ndarray
        Returns:
            dataset (Numpy ndarray): Return the data from the Dataset objects as a Numpy ndarray
        """
        df = to_numpy(self._data)
        return df

    def to_dask(self) -> ddDataframe:
        """
        Converts the Dataset object to a DASK DataFrame
        Returns:
            dataset (DASK.DataFrame): Return the data from the Dataset objects as a DASK DataFrame
        """
        return to_dask(self._data)

    def value_counts(self, col: str, compute=True) -> Union[ddSeries, pdSeries]:
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
        value_counts = self._data[col].value_counts()
        if compute:
            return value_counts.compute()
        return value_counts

    def uniques(self, col: str, approx=True, delayed=False) -> Union[int, ddScalar]:
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
        "Calculates the (exact/approximate) number of unique values in a column."
        # TODO: Enable parallel .compute() if called on multiple columns.

        # Priorities:
        # 0. only store value if exact. override approx if exact is available.
        # 1. return if already calculated
        # 2. leverage _n_uniques pre-computation if available.
        # 3. if approximate, return and skip storage

        if col not in self._n_uniques:
            if approx:
                n_uniques = self._data[col].nunique_approx()
            else:
                n_uniques = self._data[col].nunique()

            if not delayed:
                n_uniques = int(n_uniques.compute())
                self._n_uniques[col] = n_uniques

        else:
            n_uniques = self._n_uniques[col]
        return n_uniques

    def _filter_dropped_columns_values(self, columns: Union[str, list]):
        columns = columns if isinstance(columns, list) else [columns]
        self._filter_uniques(columns)
        self._filter_schema(columns)
        self._filter_missings(columns)
        self._filter_dtypes_count(columns)

    def _filter_uniques(self, columns: list):
        """Filter columns from uniques."""
        self._n_uniques = {
            k: v
            for k, v in self._n_uniques.items()
            if k not in columns
        }

    def _filter_schema(self, columns: list):
        """Filter columns from schema."""
        self._schema = {k: v for k, v in self._schema.items()
                        if k not in columns}

    def _filter_missings(self, columns: list):
        """Filter columns from missings."""
        if self._missings is None:
            return
        elif isinstance(self._missings, ddSeries):
            self._missings = None
        else:
            columns = columns if isinstance(columns, list) else [columns]
            filtered = [col for col in self.columns if col not in columns]
            self._missings = self._missings.loc[filtered]

    def _filter_dtypes_count(self, columns: list):
        """Fileter columns from _dtypes_count."""
        if self._dtypes_count is None:
            return
        self._dtypes_count = {
            k: v
            for k, v in self._dtypes_count.items()
            if k not in columns
        }

    def drop_columns(
        self, columns: Union[str, list], inplace=False
    ) -> Optional[Dataset]:
        """Drops specified columns from a Dataset.

        Args:
            columns (str or list): column labels to drop
            inplace (bool): if False, return a copy. Otherwise, drop inplace and return None.
        """
        # Validate wether the columns exist
        if inplace:
            self._data = self._data.drop(columns=columns)
            self._filter_dropped_columns_values(columns)
        else:
            return Dataset(self._data.drop(columns=columns))

    def select_columns(self, columns: Union[str, list], copy=True) -> Dataset:
        """Returns a Dataset containing only a subset with the specified columns.
        If columns is a single feature, returns a Dataset with a single column.

        Args:
            columns (str or list): column labels to select
            copy (bool): if True, return a copy. Otherwise, select inplace and return self.
        """
        columns = columns if isinstance(columns, list) else [columns]

        #validate the provided inputs columns
        if not all(e in list(self._data.columns) for e in columns):
            aux = set(columns) - set(self._data.columns)
            raise DatasetAssertionError(f"The columns {aux} are missing from the Dataset. "
                                          f"Please check your selected columns input and try again.")

        data = self._data[columns]

        if copy:
            schema = {k: v for k, v in self._schema.items()
                      if k in columns}
            return Dataset(data, schema=schema)
        else:
            dropped_cols = [col for col in self.columns if col not in columns]
            self._data = data
            self._filter_dropped_columns_values(dropped_cols)
            return self

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
        return Dataset(self._data.query(query).copy())

    def _sample_fraction(self, sample: int) -> float:
        """Sample either deterministic number of rows (exact, slow) or
        percentage of total (approximate, fast).

        Dask Dataframes API requires fraction sampling, so we convert into percentage of total if exact number of rows are requested.

        Usage:
            >>> data._sample_fraction(sample=0.01)
            0.01
            >>> data.nrows, data._sample_fraction(sample=10)
            20, 0.5

        Args:
            sample (Union[float, int]): exact number of rows or percentage of total
            nrows (int, optional): number of rows if already calculated.

        Returns:
            calc_sample (float): applicable percentage to sample from dataset.
        """
        rows = self.nrows
        if sample >= rows:
            # size is either provided (total_rows) or calculated (nrows(df))
            return 1
        elif 1 < sample < rows:  # if pct of total
            return sample / self.nrows
        else:
            raise InvalidDatasetSample(f"Requested sample ({sample}) is not valid. Please provide a sample>1.")

    def sample(
        self,
        size: Union[float, int],
        strategy: Literal["random", "stratified"] = "random",
        **strategy_params,
    ) -> Dataset:
        """
        Generate a sampled subset of the dataset.

        This method returns a sample from the dataset using either **random sampling**
        or **stratified sampling**. The sample size can be defined as an absolute number
        of rows or as a fraction of the dataset.

        Args:
            size (Union[float, int]): size (number of rows) of the sampled subset
            strategy (str["random", "stratified"]): strategy used to generate a sampled subset

        Returns:
            dataset (Dataset): the sampled subset of the dataset.
        """
        from ydata.utils.sampling.random import RandomSplitSampler
        from ydata.utils.sampling.stratified import StratifiedSampler

        strategies = {"random": RandomSplitSampler,
                      "stratified": StratifiedSampler}

        if isinstance(size, float):
            assert 0 < size < 1, f"Requested sample size ({size}) is not valid."
            frac = size
        else:
            frac = self._sample_fraction(size)

        sampler = strategies[strategy](**strategy_params)
        return sampler.sample(self, frac=frac)

    def reorder_columns(self, columns: List[str]) -> Dataset:
        """Defines the order of the underlying data based on provided 'columns'
        list of column names.

        Usage:
            >>> data.columns
            ['colA', 'colB', colC']
            >>> data.reorder_columns(['colB', 'colC']).columns
            ['colB', 'colC']
        """
        return Dataset(self._data.loc[:, columns])

    @property
    def divisions(self) -> tuple:
        """
        A property that returns the number of divisions set for the Dataset.
        Returns:
            divisions (tuple): the number of divisions set for the Dataset.
        """
        return self._data.divisions

    @property
    def known_divisions(self) -> bool:
        return self._data.known_divisions

    def sort_values(self, by: List[str], ignore_index: bool = True, inplace: bool = False) -> Optional[Dataset]:
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
        data = self._data.sort_values(by=by, ignore_index=ignore_index)
        if inplace:
            self._data = data
        else:
            return Dataset(data)

    def sorted_index(self, by: List[str]) -> pdSeries:
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
        return self._data[by] \
            .repartition(npartitions=1).reset_index(drop=True) \
            .sort_values(by=by).compute() \
            .index.to_frame(index=False).iloc[:, 0]

    def head(self, n=5) -> pdDataframe:
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
        head_df = self._data.head(n, npartitions=1, compute=True)
        if head_df.shape[0] < n:
            head_df = self._data.head(n, npartitions=-1, compute=True)
        return head_df

    def tail(self, n=5) -> pdDataframe:
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
        return self._data.tail(n, compute=True)

    ##################
    # Dunder methods #
    ##################
    def __len__(self) -> int:
        """Implements utility to call len(Dataset) directly, returning the
        number of rows.

        Usage:
        >>> len(data)
        """
        return self.nrows

    def __contains__(self, key) -> bool:
        """True if key is in Dataset columns.

        Usage:
        >>> 'my_column' in data
        """
        return key in self.columns

    def __getitem__(self, key) -> Dataset:
        """
        Usage:
        >>> data[ ['columnA', 'columnB'] ]
        """
        return self.select_columns(key)

    def _build_repr(self) -> dict:
        dataset = {
            "Shape": self.shape(lazy_eval=False, delayed=False),
            "Schema": pdDataframe(
                [
                    {"Column": k, "Variable type": v.vartype.value}
                    for k, v in self._schema.items()
                ]
            ),
        }
        return dataset

    def __str__(self) -> str:
        """Dunder method to pretty print the content of the object Dataset."""
        pretty_summary = self._build_repr()
        str_repr = TextStyle.BOLD + "Dataset \n \n" + TextStyle.END
        for k, val in pretty_summary.items():
            str_repr += TextStyle.BOLD + f"{k}: " + TextStyle.END
            if type(val) != pdDataframe:
                str_repr += str(val)
            else:
                str_repr += "\n"
                str_repr += val.to_string() + "\n"
            str_repr += "\n"
        return str_repr


def _astype(data: ddDataframe, column: str, vartype: str, format: str | None = None) -> ddDataframe:
    if vartype in ["date", "time", "datetime"]:
        data[column] = dd.to_datetime(
            data[column], format=format, errors="coerce").dt.tz_localize(None)
    else:
        data = data.astype({column: vartype})

        if format is not None:
            warn("Parameter 'format' is valid only for vartype='datetime'")

    return data
