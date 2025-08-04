import abc
from abc import abstractmethod
from ydata.connectors.base_connector import BaseConnector
from ydata.dataset import Dataset
from ydata.dataset.filetype import FileType

class ObjectStorageConnector(BaseConnector, metaclass=abc.ABCMeta):
    """Generic connector for object-oriented storages (e.g. S3, GCS, Azure
    BLOB)"""
    def __init__(self) -> None: ...
    @abstractmethod
    def parse_connector_url(self, url: str):
        """Parses and validates a connector-specific url."""
    @abstractmethod
    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        """Get connector-specific file paths."""
    @property
    def filesystem(self) -> None:
        """Get connector-specific file paths."""
    @property
    def client(self):
        """Getter method for client property. Lazy-eval: create only if does not exist."""
    @abstractmethod
    def set_client(self):
        """Sets a new client for the Connector."""
    def read_file(self, path: str, file_type: FileType = ..., sample_size: int | None = None, sep: str = ',', has_header: bool = True, columns: list[str] | None = None, schema: dict | None = None, *args, **kwargs) -> Dataset:
        """Reads a file from a specified path. All the args and kwargs are
        down-propagated to read method.

        Args:
            path (str): Filepath for the file/s to be read
            file_type (FileType, optional):  if provided, cast to FileType class.
            sample_size (int, optional): defines the number of rows requested when sampling from the dataset
            has_header (bool): informs if the dataset has a header row or not
            columns (Optional[List[str]]): provide list to replace existing or automatically provided column names

        Returns:
            df (Dataset): The file or sample of the file (if sample_size was passed)
        """
    def read_sample(self, path: str, file_type: FileType = ..., sample_size: int = ..., sep: str = ',', *args, **kwargs) -> Dataset:
        """Reads a sample of a file for a given path.

        Similar to read_file with sampling.
        """
    def write_file(self, data, path: str, file_type: FileType | None = None, *args, **kwargs): ...
    @abstractmethod
    def test(self): ...
