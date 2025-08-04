"""Object Storage Connector."""
from abc import abstractmethod
from typing import List, Optional
from urllib import parse

from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.exceptions import DataConnectorsException
from ydata.dataset.filetype import FileType, infer_file_type
from ydata.connectors.logger import logger
from ydata.connectors.readers import DaskReader
from ydata.connectors.storages import _MAX_SAMPLE
from ydata.connectors.utils import get_schema, nsample
from ydata.connectors.utils.header import validate_header
from ydata.dataset import Dataset
from ydata.dataset.engines import to_dask


class ObjectStorageConnector(BaseConnector):
    """Generic connector for object-oriented storages (e.g. S3, GCS, Azure
    BLOB)"""

    def __init__(self):
        super().__init__()
        self._client = None
        self._filesystem = None

    def _cast_filetype(self, path: str, file_type: FileType):
        "Infer type from path if file_type not given and cast to FileType class."
        file_type = file_type or infer_file_type(path)
        return self.parse_file_type(file_type)

    @abstractmethod
    def parse_connector_url(self, url: str):
        "Parses and validates a connector-specific url."

    @abstractmethod
    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        "Get connector-specific file paths."

    @property
    def filesystem(self):
        "Get connector-specific file paths."

    @property
    def client(self):
        "Getter method for client property. Lazy-eval: create only if does not exist."
        if self._client is None:
            self.set_client()
        return self._client

    @abstractmethod
    def set_client(self):
        "Sets a new client for the Connector."

    def _check_path(self, path: str, file_type: FileType):
        "Internal method to validate provided path."
        # Applies connector-specific parser
        bucket, filename = self.parse_connector_url(path)

        # Checks file extension
        extension = self.check_file_extension(filename)

        # Retrieves the file paths.
        file_paths = self.get_file_paths(
            path=path, file_type=file_type, extension=extension
        )

        return file_paths, bucket, filename

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
        # Guarantees FileType is correctly casted.
        fileType = self._cast_filetype(path=path, file_type=file_type)
        file_paths, _, _ = self._check_path(path=path, file_type=fileType)


        if file_type == FileType.PARQUET:
            logger.warning(
                f'The parameters "sep", "has_header" and "schema" are only considered for f{FileType.CSV} files.'
            )
        elif file_type == FileType.CSV:
            # Schema calculation is only available for csv files
            # Parquet files have a different strategy to deal with data types
            if schema is None:
                schema = get_schema(
                    file_paths[0],
                    file_type,
                    sep=sep,
                    storage_options=self.storage_options,
                )


        # Add here the read_csv and read_parquet logic for the schema
        try:
            df = DaskReader(fileType).read(
                file_paths,
                sep=sep,
                *args,
                storage_options=self.storage_options,
                dtype=schema,
                **kwargs,
            )

            # Validate the provided header prior any Dataset computation
            replace, header, header_names = validate_header(
                has_header, header_cols=list(df.columns), columns=columns
            )

        except FileNotFoundError:
            raise DataConnectorsException(
                "The file you have provided does not exist.")

        except Exception as exc:
            raise DataConnectorsException(
                "There was an error reading the path provided."
            ) from exc

        if replace:
            # replace column names by the new one's defined. This is required as DASK does not support skipnrows as Pandas Dataframe
            df = df.rename(columns=dict(zip(df.columns, header_names)))

        # Remove unnamed default column
        df = self._rm_unnamed(df)

        # frac cannot be None
        if sample_size:
            df = nsample(df, sample_size)
        return Dataset(df)

    def read_sample(
        self, path: str, file_type: FileType = FileType.CSV, sample_size: int = _MAX_SAMPLE, sep=",", *args, **kwargs
    ) -> Dataset:
        """Reads a sample of a file for a given path.

        Similar to read_file with sampling.
        """
        return self.read_file(path=path, file_type=file_type, sample_size=sample_size, sep=sep, *args, **kwargs)

    def write_file(
        self, data, path: str, file_type: Optional[FileType] = None, *args, **kwargs
    ):
        file_type = FileType(file_type) if file_type else file_type

        """Writes data into a given path."""
        if file_type == FileType.PARQUET:
            logger.warning(
                'Some parameters are only considered for CSV files: "sep", "has_header" and "columns".'
            )

        # Ensures correctness of URL
        _, _ = self.parse_connector_url(path)
        scheme = parse.urlparse(path).scheme
        single_file_default = True if scheme == '' else False

        # Enforces the correct filetype
        fileType = self._cast_filetype(path=path, file_type=file_type)

        # Enforce a Dask Dataframe
        dataset = data.to_dask() if isinstance(data, Dataset) else to_dask(data)

        try:
            if fileType == FileType.PARQUET:
                write_index = kwargs.get("write_index", False)
                dataset.to_parquet(
                    path, write_index=write_index, storage_options=self.storage_options
                )
            elif fileType == FileType.CSV:
                # if not schema specified, the file is written locally and can be in a single file
                single_file = kwargs.get("single_file", single_file_default)
                index = kwargs.get("index", False)
                dataset.to_csv(
                    path,
                    index=index,
                    single_file=single_file,
                    storage_options=self.storage_options,
                )
            else:
                raise ValueError(f"Not allowed file type: {fileType.name}")
        except Exception as exc:
            exc_message = f"\n\tFull exception message: {str(exc)}." if str(
                exc) else ""
            raise DataConnectorsException(
                f"It was not possible to write the dataset in the given path ({path})."
                + exc_message
            ) from exc

    @abstractmethod
    def test(self):
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(extraglobs={"obs_connector": ObjectStorageConnector()})
