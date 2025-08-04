from _typeshed import Incomplete
from typing import NamedTuple
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.dataset.filetype import FileType as FileType

class GCSSpec(NamedTuple('GCSSpec', [('bucket', Incomplete), ('blob', Incomplete)])):
    """A specification for gcs configuration."""

def parse_gcs_path(gcs_path):
    """Parses and validates a google cloud storage url.

    Returns:
        tuple(bucket_name, blob).
    """

class GCSConnector(ObjectStorageConnector):
    """Google Cloud Storage connector.

    Based on Dask to handle large data volumes.
    """
    STORAGE_TYPE: Incomplete
    bucket_name: Incomplete
    credentials: Incomplete
    storage_options: Incomplete
    def __init__(self, project_id, gcs_credentials: Incomplete | None = None, key_path: Incomplete | None = None, keyfile_dict: Incomplete | None = None, **kwargs) -> None: ...
    def set_client(self) -> None:
        """Sets a new gc client.

        Returns:
            Service client instance
        """
    @property
    def filesystem(self):
        """
        Create a filesystem conenction to a certain bucket
        Args:
            bucket_name: `str`.
        Returns:
            A filesystem connection
        """
    def set_filesystem(self, bucket_name) -> None:
        """Defines the GCS filesystem."""
    def set_env_vars(self) -> None:
        """Defines the environment variables."""
    def parse_connector_url(self, url: str):
        """Parses and validates a google cloud storage url.

        Creates the Filesystem connection.
        Returns:
            tuple(bucket_name, blob).
        """
    def get_blob(self, blob, bucket_name: Incomplete | None = None):
        """Get a file in Google Cloud Storage.

        Args:
            blob: `str`. the path to the object to check in the Google cloud storage bucket.
            bucket_name: `str`. Name of the bucket in which the file is stored
        """
    def get_bucket(self, bucket_name):
        """Gets a bucket by name.

        Args:
            bucket_name: `str`. Name of the bucket
        """
    def check_blob(self, blob_name: str, bucket_name: str | None = None):
        """Checks for the existence of a file in Google Cloud Storage.

        Args:
            blob_name: `str`. the path to the object to check in the Google cloud storage bucket.
            bucket_name: `str`. Optional. Name of the bucket in which the file is stored
        """
    def delete_blob_if_exists(self, blob_name: str, bucket_name: str | None = None):
        """Deletes a BLOB from GCS if exists"""
    def ls(self, path: str):
        """Lists files and dirs."""
    def list(self, key, bucket_name: Incomplete | None = None, path: Incomplete | None = None, delimiter: str = '/', blobs: bool = True, prefixes: bool = True):
        """List prefixes and blobs in a bucket.

        Args:
            key: `str`. a key prefix.
            bucket_name: `str`. the name of the bucket.
            path: `str`. an extra path to append to the key.
            delimiter: `str`. the delimiter marks key hierarchy.
            blobs: `bool`. if it should include blobs.
            prefixes: `bool`. if it should include prefixes.
        Returns:
             Service client instance
        """
    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        """Given a path, return the valid files for a given file_type."""
    def test(self) -> None: ...
