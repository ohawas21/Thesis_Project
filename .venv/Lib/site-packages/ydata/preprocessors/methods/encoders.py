"""
    Encoders logic
"""
from typing import Union, Optional, TypeVar

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from dask.dataframe import DataFrame, Series, from_dask_array
from dask.array import Array, blockwise

DataFrameType = Union[pd.DataFrame, DataFrame]
ArrayLike = TypeVar("ArrayLike", Array, np.ndarray)
SeriesType = Union[Series, pd.Series]

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal (integer) encode categorical columns.

    Parameters
    ----------
    columns : sequence, optional
        The columns to encode. Must be categorical dtype.
        Encodes all categorical dtype columns by default.

    Attributes
    ----------
    columns_ : Index
        The columns in the training data before/after encoding

    categorical_columns_ : Index
        The categorical columns in the training data

    noncategorical_columns_ : Index
        The rest of the columns in the training data

    dtypes_ : dict
        Dictionary mapping column name to either

        * instances of CategoricalDtype (pandas >= 0.21.0)
        * tuples of (categories, ordered)

    Notes
    -----
    This transformer only applies to dask and pandas DataFrames. For dask
    DataFrames, all of your categoricals should be known.

    The inverse transformation can be used on a dataframe or array.

    Examples
    --------
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4],
    ...                      "B": pd.Categorical(['a', 'a', 'a', 'b'])})
    >>> enc = OrdinalEncoder()
    >>> trn = enc.fit_transform(data)
    >>> trn
       A  B
    0  1  0
    1  2  0
    2  3  0
    3  4  1

    >>> enc.columns_
    Index(['A', 'B'], dtype='object')

    >>> enc.non_categorical_columns_
    Index(['A'], dtype='object')

    >>> enc.categorical_columns_
    Index(['B'], dtype='object')

    >>> enc.dtypes_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}

    >>> enc.fit_transform(dd.from_pandas(data, 2))
    Dask DataFrame Structure:
                       A     B
    npartitions=2
    0              int64  int8
    2                ...   ...
    3                ...   ...
    Dask Name: assign, 8 tasks

    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(
        self, X: DataFrameType, y: Optional[Union[ArrayLike, SeriesType]] = None
    ) -> "OrdinalEncoder":
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        columns = self.columns
        if columns is None:
            columns = X.select_dtypes(include=["category"]).columns
        else:
            for column in columns:
                assert isinstance(
                    X[column].dtype, pd.CategoricalDtype
                ), "Must be categorical"

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)

        self.dtypes_ = {col: X[col].dtype for col in self.categorical_columns_}

        return self

    def transform(
        self, X: DataFrameType, y: Optional[Union[ArrayLike, SeriesType]] = None
    ) -> DataFrameType:
        """Ordinal encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns)
            )
        if not isinstance(X, (pd.DataFrame, DataFrame)):
            raise TypeError("Unexpected type {}".format(type(X)))

        X = X.copy()
        for col in self.categorical_columns_:
            X[col] = X[col].cat.codes
        return X

    def inverse_transform(
        self, X: Union[ArrayLike, DataFrameType]
    ) -> Union[ArrayLike, DataFrameType]:
        """Inverse ordinal-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)

        elif isinstance(X, Array):
            # later on we concat(..., axis=1), which requires
            # known divisions. Suboptimal, but I think unavoidable.
            unknown = np.isnan(X.chunks[0]).any()
            if unknown:
                lengths = blockwise(len, "i", X[:, 0], "i", dtype="i8").compute()
                X = X.copy()
                chunks: tuple = (tuple(lengths), X.chunks[1])
                X._chunks = chunks

            X = from_dask_array(X, columns=self.columns_)

        big = isinstance(X, DataFrame)

        if big:
            divisions = np.array(X.divisions)
            divisions[-1] = divisions[-1] + 1
            chunks = tuple(divisions[1:] - divisions[:-1])

        X = X.copy()
        for col in self.categorical_columns_:
            dtype = self.dtypes_[col]
            categories, ordered = dtype.categories, dtype.ordered

            # use .values to avoid warning from pandas
            codes = X[col].values

            if big:
                # dask
                codes._chunks = (chunks,)
                # Need a Categorical.from_codes for dask
                series = (
                    from_dask_array(codes, columns=col)
                    .astype("category")
                    .cat.set_categories(np.arange(len(categories)), ordered=ordered)
                    .cat.rename_categories(categories)
                )
                # Bug in pandas <= 0.20.3 lost name
                if series.name is None:
                    series.name = col
            else:
                # pandas
                series = pd.Series(
                    pd.Categorical.from_codes(codes, categories, ordered=ordered),
                    name=col,
                )

            X[col] = series

        return X
