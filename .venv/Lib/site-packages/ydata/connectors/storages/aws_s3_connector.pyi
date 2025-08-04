from _typeshed import Incomplete
from typing import NamedTuple
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.dataset.filetype import FileType as FileType

class S3Spec(NamedTuple('S3Spec', [('bucket', Incomplete), ('key', Incomplete)])):
    """A specification for s3 configuration."""

def parse_s3_path(s3_path):
    """Parses and validates an S3 url.

    Returns:
         tuple(bucket_name, key).
    """

class S3Connector(ObjectStorageConnector):
    """S3 store service using Modian."""
    STORAGE_TYPE: Incomplete
    credentials: Incomplete
    storage_options: Incomplete
    def __init__(self, access_key_id, secret_access_key, aws_ssl: bool = True, session_token: Incomplete | None = None, aws_region: Incomplete | None = None, ds: Incomplete | None = None, **kwargs) -> None: ...
    def set_client(self) -> None:
        """Sets a new s3 boto3 client.

        Args:
            endpoint_url: `str`. The complete URL to use for the constructed client.
            aws_access_key_id: `str`. The access key to use when creating the client.
            aws_secret_access_key: `str`. The secret key to use when creating the client.
            aws_session_token: `str`. The session token to use when creating the client.
            region_name: `str`. The name of the region associated with the client.
                A client is associated with a single region.
        Returns:
            Service client instance
        """
    def set_env_vars(self) -> None: ...
    @property
    def filesystem(self):
        """Gets or sets the S3 Filesystem based on the provided credentials."""
    def set_filesystem(self, credentials) -> None:
        """
        Sets a new s3fs S3FileSystem client
        Args:
            credentials that include the following properties
                - endpoint_url: `str`. The complete URL to use for the constructed client.
                - aws_access_key_id: `str`. The access key to use when creating the client.
                - aws_secret_access_key: `str`. The secret key to use when creating the client.
                - aws_session_token: `str`. The session token to use when creating the client.
                - region_name: `str`. The name of the region associated with the client.
                                      A client is associated with a single region.
        Returns:
            Service client instance
        """
    @property
    def resource(self):
        """Gets or sets a Boto3 Resource Service based on the provided credentials."""
    def set_resource(self, **kwargs) -> None:
        """Sets a new s3 boto3 resource.

        Args:
            endpoint_url: `str`. The complete URL to use for the constructed client.
            aws_access_key_id: `str`. The access key to use when creating the client.
            aws_secret_access_key: `str`. The secret key to use when creating the client.
            aws_session_token: `str`. The session token to use when creating the client.
            region_name: `str`. The name of the region associated with the client.
                A client is associated with a single region.
        Returns:
             Service resource instance
        """
    def parse_connector_url(self, url: str):
        """Parses and validates an S3 url.

        Returns:
             tuple(bucket_name, key).
        """
    @staticmethod
    def check_prefix_format(prefix: str, delimiter: str): ...
    def check_bucket(self, bucket_name):
        """Checks if a bucket exists.

        Args:
            bucket_name: `str`. Name of the bucket
        """
    def get_bucket(self, bucket_name):
        """Gets a bucket by name.

        Args:
            bucket_name: `str`. Name of the bucket
        """
    def get_key(self, key, bucket_name: Incomplete | None = None):
        """
        Returns a boto3.s3.Object
        Args:
            key: `str`. the path to the key.
            bucket_name: `str`. the name of the bucket.
        """
    def read_key(self, key, bucket_name: Incomplete | None = None):
        """
        Reads a key from S3
        Args:
            key: `str`. S3 key that will point to the file.
            bucket_name: `str`. Name of the bucket in which the file is stored.
        """
    def ls(self, path): ...
    def list(self, bucket_name, prefix: str = '', delimiter: str = '/', page_size: Incomplete | None = None, max_items: Incomplete | None = None, keys: bool = True, prefixes: bool = True):
        """Lists prefixes and contents in a bucket under prefix.

        Args:
            bucket_name: `str`. the name of the bucket
            prefix: `str`. a key prefix
            delimiter: `str`. the delimiter marks key hierarchy.
            page_size: `str`. pagination size
            max_items: `int`. maximum items to return
            keys: `bool`. if it should include keys
            prefixes: `boll`. if it should include prefixes
        """
    def list_keys(self, bucket_name, prefix: str = '', delimiter: str = '', page_size: Incomplete | None = None, max_items: Incomplete | None = None):
        """
        Lists keys in a bucket under prefix and not containing delimiter
        Args:
            bucket_name: `str`. the name of the bucket
            prefix: `str`. a key prefix
            delimiter: `str`. the delimiter marks key hierarchy.
            page_size: `int`. pagination size
            max_items: `int`. maximum items to return
        """
    def check_key(self, key, bucket_name: Incomplete | None = None):
        """
        Checks if a key exists in a bucket
        Args:
            key: `str`. S3 key that will point to the file
            bucket_name: `str`. Name of the bucket in which the file is stored
        """
    def get_file_paths(self, path: str, file_type: FileType, extension: str): ...
    def test(self) -> None: ...
