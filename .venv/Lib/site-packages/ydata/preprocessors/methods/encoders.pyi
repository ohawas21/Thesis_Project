import numpy as np
from _typeshed import Incomplete
from dask.array import Array
from sklearn.base import BaseEstimator, TransformerMixin
from typing import TypeVar

DataFrameType: Incomplete
ArrayLike = TypeVar('ArrayLike', Array, np.ndarray)
SeriesType: Incomplete

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    '''Ordinal (integer) encode categorical columns.

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
    ...                      "B": pd.Categorical([\'a\', \'a\', \'a\', \'b\'])})
    >>> enc = OrdinalEncoder()
    >>> trn = enc.fit_transform(data)
    >>> trn
       A  B
    0  1  0
    1  2  0
    2  3  0
    3  4  1

    >>> enc.columns_
    Index([\'A\', \'B\'], dtype=\'object\')

    >>> enc.non_categorical_columns_
    Index([\'A\'], dtype=\'object\')

    >>> enc.categorical_columns_
    Index([\'B\'], dtype=\'object\')

    >>> enc.dtypes_
    {\'B\': CategoricalDtype(categories=[\'a\', \'b\'], ordered=False)}

    >>> enc.fit_transform(dd.from_pandas(data, 2))
    Dask DataFrame Structure:
                       A     B
    npartitions=2
    0              int64  int8
    2                ...   ...
    3                ...   ...
    Dask Name: assign, 8 tasks

    '''
    columns: Incomplete
    def __init__(self, columns: Incomplete | None = None) -> None: ...
    columns_: Incomplete
    categorical_columns_: Incomplete
    non_categorical_columns_: Incomplete
    dtypes_: Incomplete
    def fit(self, X: DataFrameType, y: ArrayLike | SeriesType | None = None) -> OrdinalEncoder:
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
    def transform(self, X: DataFrameType, y: ArrayLike | SeriesType | None = None) -> DataFrameType:
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
    def inverse_transform(self, X: ArrayLike | DataFrameType) -> ArrayLike | DataFrameType:
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
