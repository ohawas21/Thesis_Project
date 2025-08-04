"""Implementation of reader methods depending on the framework (e.g. Dask,
Pandas)."""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

from abc import ABC

import dask.dataframe as dd
import pandas as pd

from ydata.dataset.filetype import FileType


class ReaderException(Exception):
    "Exception raised during implementation of Reader methods."


class AbstractReader(ABC):
    """Abstract definition for the relationship between a FileType and reader
    methods."""

    # pylint: disable=too-few-public-methods,invalid-name

    def __init__(self, file_type: FileType):
        self.file_type = file_type

    @property
    def read(self):
        "Returns the correct method to read the data according to the FileType."
        try:
            return self.__FILETYPE_READER_MAP__[self.file_type]
        except KeyError as kexc:
            # If key is not found in reader implementation dictionary __FILETYPE_READER_MAP__,
            # file type is not supported.
            raise ReaderException(
                f"File type {self.file_type.name} is not supported with {self.__class__.__name__}."
            ) from kexc
        except Exception as exc:
            raise ReaderException(
                f"There was an error reading data from {self.file_type} with {self.__class__.__name__}."
            ) from exc


class DaskReader(AbstractReader):
    """Dask specific reader methods.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> reader = DaskReader(FileType.CSV)
        >>> reader.read(*args, **kwargs)
    """

    # pylint: disable=too-few-public-methods

    __FILETYPE_READER_MAP__ = {
        FileType.PARQUET: dd.read_parquet,
        FileType.CSV: dd.read_csv,
    }


class PandasReader(AbstractReader):
    """Pandas specific reader methods.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> reader = PandasReader(FileType.PARQUET)
        >>> reader.read(*args, **kwargs)
    """

    # pylint: disable=too-few-public-methods

    __FILETYPE_READER_MAP__ = {
        FileType.PARQUET: pd.read_parquet,
        FileType.CSV: pd.read_csv,
    }
