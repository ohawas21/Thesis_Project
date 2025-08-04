from _typeshed import Incomplete
from abc import ABC
from ydata.dataset.filetype import FileType

class ReaderException(Exception):
    """Exception raised during implementation of Reader methods."""

class AbstractReader(ABC):
    """Abstract definition for the relationship between a FileType and reader
    methods."""
    file_type: Incomplete
    def __init__(self, file_type: FileType) -> None: ...
    @property
    def read(self):
        """Returns the correct method to read the data according to the FileType."""

class DaskReader(AbstractReader):
    """Dask specific reader methods.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> reader = DaskReader(FileType.CSV)
        >>> reader.read(*args, **kwargs)
    """
    __FILETYPE_READER_MAP__: Incomplete

class PandasReader(AbstractReader):
    """Pandas specific reader methods.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> reader = PandasReader(FileType.PARQUET)
        >>> reader.read(*args, **kwargs)
    """
    __FILETYPE_READER_MAP__: Incomplete
