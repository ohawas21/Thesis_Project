from _typeshed import Incomplete
from ydata.connectors.base_connector import BaseConnector
from ydata.dataset import Dataset
from ydata.dataset.filetype import FileType

class LocalConnector(BaseConnector):
    """Local Storage connector.

    Based on Dask to handle large data volumes.
    """
    STORAGE_TYPE: Incomplete
    storage_options: Incomplete
    def __init__(self) -> None: ...
    def ls(self, path: str) -> dict[str, list[str]]:
        """Returns a list of subdirectories and files contained in a given
        relative path.

        Args:
            path: Relative path where to start search
        Returns:
            Dictionary with lists of all files or directories under the path after the check.
        """
    def list(self, path: str, abs_path: bool = False) -> dict[str, list[str]]:
        """Returns a list of subdirectories and files contained in a given
        relative or absolute path.

        Args:
            path: Path where to start search
            abs_path:
        Returns:
            Dictionary with lists of all files or directories under the path after the check.
        """
    def get_file_paths(self, path: str, file_type: FileType, extension: str) -> list[str]:
        """Given a path, return the valid files for a given file_type."""
    def read_sample(self, path: str, file_type: FileType = ..., sample_size: int = ..., sep: str = ',', *args, **kwargs) -> Dataset:
        """Reads a sample of a file for a given path.

        Similar to read_file with sampling.
        """
    def read_file(self, path: str, file_type: FileType = ..., sample_size: int | None = None, sep: str = ',', has_header: bool = True, columns: list[str] | None = None, schema: dict | None = None, *args, **kwargs) -> Dataset:
        """Reads a file from a specified path.

        All the args and kwargs are down-propagated to read method.
        Args:
            path (str): Filepath for the file/s to be read
            file_type (FileType, optional):  if provided, cast to FileType class.
            sample_size (Union[float, int], optional): defines the fraction for sampling.
            has_header (bool): informs if the dataset has a header row or not
            columns (Optional[List[str]]): provide list to replace existing or automatically provided column names
        Returns:
            df (Dataset):
        """
    def write_file(self, data, path: str, file_type: FileType | None = None, *args, **kwargs):
        """Writes data into a given path."""
    def test(self) -> None: ...
