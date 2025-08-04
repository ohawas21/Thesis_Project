"Connector for Local Storage."
from __future__ import absolute_import, division, print_function

import os
from typing import Dict, List, Optional

from numpy import dtype
from pandas import concat, read_csv, read_parquet

from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.exceptions import DataConnectorsException
from ydata.dataset.filetype import FileType, infer_file_type
from ydata.connectors.logger import logger
from ydata.connectors.storages import _LOCAL_STORAGE, _MAX_SAMPLE
from ydata.connectors.utils import get_schema
from ydata.connectors.utils.header import validate_header
from ydata.dataset import Dataset
from ydata.dataset.engines import to_dask


class LocalConnector(BaseConnector):
    """Local Storage connector.

    Based on Dask to handle large data volumes.
    """

    STORAGE_TYPE = _LOCAL_STORAGE

    def __init__(self):
        # Bypass dask client by not calling BaseConnector.__init__
        self.storage_options = None

    def _cast_filetype(self, path: str, file_type: FileType):
        "Infer type from path if file_type not given and cast to FileType class."
        file_type = file_type or infer_file_type(path)
        return self.parse_file_type(file_type)

    # pylint: disable=C0103
    def ls(self, path: str) -> Dict[str, List[str]]:
        """Returns a list of subdirectories and files contained in a given
        relative path.

        Args:
            path: Relative path where to start search
        Returns:
            Dictionary with lists of all files or directories under the path after the check.
        """
        return self.list(path=path)

    def list(self, path: str, abs_path: bool = False) -> Dict[str, List[str]]:
        """Returns a list of subdirectories and files contained in a given
        relative or absolute path.

        Args:
            path: Path where to start search
            abs_path:
        Returns:
            Dictionary with lists of all files or directories under the path after the check.
        """

        def list_dirs():
            return self._list(path, os.path.isdir, abs_path)

        def list_files():
            matches = self._list(path, os.path.isfile, abs_path)
            if abs_path:
                return [(f, os.path.getsize(f)) for f in matches]
            return [(f, os.path.getsize(os.path.join(path, f))) for f in matches]

        return {"dirs": list_dirs(), "files": list_files()}

    def get_file_paths(
        self, path: str, file_type: FileType, extension: str
    ) -> List[str]:
        "Given a path, return the valid files for a given file_type."
        if extension is None:
            files = os.listdir(path)
            file_paths = [
                os.path.join(path, file)
                for file in files
                if self.check_file_extension(file) == file_type.value
            ]
        else:
            file_paths = [path]
        return file_paths

    def read_sample(
        self, path: str, file_type: FileType = FileType.CSV, sample_size: int = _MAX_SAMPLE, sep=",", *args, **kwargs
    ) -> Dataset:
        """Reads a sample of a file for a given path.

        Similar to read_file with sampling.
        """
        return self.read_file(path=path, file_type=file_type, sample_size=sample_size, sep=sep, *args, **kwargs)

    def read_file(
        self,
        path: str,
        file_type: FileType = FileType.CSV,
        sample_size: int | None = None,
        sep: str = ",",
        has_header: bool = True,
        columns: Optional[List[str]] = None,
        schema: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> Dataset:
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
        # Guarantees FileType is correctly casted.
        fileType = self._cast_filetype(path=path, file_type=file_type)

        # Check wether the provided path is valid
        if os.path.isdir(path):
            files = self._list(path, check=os.path.isfile, abs_path=True)
            files = [file for file in files if fileType.value in file]
        else:
            files = [path]

        if FileType(file_type) == FileType.PARQUET:
            logger.warning(
                f'The parameters "sep", "has_header" and "schema" are only considered for {FileType.CSV} files.'
            )
        elif FileType(file_type) == FileType.CSV:
            # Schema calculation is only available for csv files
            # Parquet files have a different strategy to deal with data types
            if schema is None:
                schema = get_schema(files[0], file_type, sep=sep)

        try:
            ls_data = []
            for file in files:
                try:
                    if fileType == FileType.CSV:
                        ls_data.append(
                            read_csv(file, encoding_errors="ignore", sep=sep))
                    elif fileType == FileType.PARQUET:
                        ls_data.append(read_parquet(file))
                    else:
                        raise DataConnectorsException(
                            f'The provided file type: {file_type} is not currently supported with LocalFile connector.')
                except FileNotFoundError:
                    raise DataConnectorsException(
                        f'{file} does not exist or access not allowed.')

            df = concat(ls_data)

            if sample_size:
                _sample_size = sample_size if sample_size < df.shape[0] else df.shape[0]
                df = df.sample(n=_sample_size)

            df = self._rm_unnamed(df)

            # Validate the provided header prior any Dataset computation
            replace, _, header_names = validate_header(
                has_header, header_cols=list(df.columns), columns=columns
            )
        except FileNotFoundError:
            raise DataConnectorsException(
                "The file you have provided does not exist.")

        except Exception as exc:
            raise DataConnectorsException(
                "There was an error reading the path provided."
            ) from exc

        # Checking infered dtypes
        dtypes = dict(df.dtypes)
        # Due to the potential presence of NA values, all the integers need to be converted to floats
        for k, v in dtypes.items():
            if v == int:
                dtypes[k] = dtype("float32")

        if replace:
            # replace column names by the new one's defined. This is required as DASK does not support skipnrows as Pandas Dataframe
            df = df.rename(columns=dict(zip(df.columns, header_names)))

        return Dataset(df)

    def write_file(
        self, data, path: str, file_type: Optional[FileType] = None, *args, **kwargs
    ):
        """Writes data into a given path."""
        if file_type == FileType.PARQUET:
            logger.warning(
                'Some parameters are only considered for CSV files: "sep", "has_header" and "columns".'
            )

        # Enforce a Dask Dataframe
        dataset = data.to_dask() if isinstance(data, Dataset) else to_dask(data)

        try:
            if file_type == FileType.PARQUET:
                write_index = kwargs.get("write_index", False)
                dataset.to_parquet(
                    path, write_index=write_index, storage_options=self.storage_options
                )
            elif file_type == FileType.CSV:
                single_file = kwargs.get("single_file", True)
                index = kwargs.get("index", False)
                dataset.to_csv(
                    path,
                    index=index,
                    single_file=single_file,
                    storage_options=self.storage_options,
                )
            else:
                raise ValueError(f"Not allowed file type: {file_type.name}")
        except Exception as exc:
            exc_message = f"\n\tFull exception message: {str(exc)}." if str(
                exc) else ""
            raise DataConnectorsException(
                f"It was not possible to write the dataset in the given path ({path})."
                + exc_message
            ) from exc

    def test(self):
        pass

    @staticmethod
    def _list(path: str, check, abs_path=False) -> List[str]:
        """
        List all entities directly under 'dir_name' that satisfy 'filter_func'
        Args:
            path: path where to start search
            check: function to check whether or not to include the results.
            abs_path: If True will return results with abs path.
        Returns:
            list of all files or directories under the path after the check.
        """
        if not os.path.isdir(path):
            raise Exception("Invalid parent directory '%s'" % path)
        matches = [x for x in os.listdir(path) if check(os.path.join(path, x))]
        return [os.path.join(path, m) for m in matches] if abs_path else matches
