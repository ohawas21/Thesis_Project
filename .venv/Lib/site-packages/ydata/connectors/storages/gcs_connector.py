"""Google Cloud Storage Connector."""
from __future__ import absolute_import, division, print_function

from collections import namedtuple
from json import dump
from os import environ
from os import path as os_path
from typing import Optional
from urllib import parse

from ydata.dataset.filetype import FileType

from ydata.connectors import logger, settings
from ydata.connectors.clients import gc_client
from ydata.connectors.exceptions import DataConnectorsException, GCSPathError
from ydata.connectors.storages import _GCS_STORAGE
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.connectors.utils import create_tmp


class GCSSpec(namedtuple("GCSSpec", "bucket blob")):
    """A specification for gcs configuration."""


def parse_gcs_path(gcs_path):
    """Parses and validates a google cloud storage url.

    Returns:
        tuple(bucket_name, blob).
    """
    parsed_url = parse.urlparse(gcs_path)
    if not parsed_url.netloc:
        raise GCSPathError('Received an invalid GCS url `{}`'.format(gcs_path))
    if parsed_url.scheme != 'gs':
        raise GCSPathError('Received an invalid url GCS `{}`'.format(gcs_path))
    blob = parsed_url.path.lstrip('/')
    return GCSSpec(parsed_url.netloc, blob)


class GCSConnector(ObjectStorageConnector):
    """Google Cloud Storage connector.

    Based on Dask to handle large data volumes.
    """

    STORAGE_TYPE = _GCS_STORAGE

    def __init__(
        self,
        project_id,
        gcs_credentials=None,
        key_path=None,
        keyfile_dict=None,
        **kwargs
    ):
        super().__init__()
        self._project_id = project_id
        self._gcs_credentials = gcs_credentials
        self._key_path = key_path
        self._keyfile_dict = keyfile_dict
        self.bucket_name = None
        self._scopes = kwargs.get("scopes")

        self.credentials = {
            "project_id": self._project_id,
            "credentials": self._gcs_credentials,
            "key_path": self._key_path,
            "keyfile_dict": self._keyfile_dict,
            "scopes": self._scopes,
        }

        self.storage_options = {"token": self._keyfile_dict}

    def set_client(self):
        """Sets a new gc client.

        Returns:
            Service client instance
        """
        self._client = gc_client.get_gc_client(**self.credentials)

    @property
    def filesystem(self):
        """
        Create a filesystem conenction to a certain bucket
        Args:
            bucket_name: `str`.
        Returns:
            A filesystem connection
        """
        self.set_filesystem(self.bucket_name)
        return self._filesystem

    def set_filesystem(self, bucket_name):
        "Defines the GCS filesystem."
        self._filesystem = gc_client.get_gc_filesystem(
            self.client, bucket_name)

    def set_env_vars(self):
        "Defines the environment variables."
        if self._key_path:
            environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials["key_path"]
        elif self.credentials["keyfile_dict"]:
            create_tmp()
            with open(settings.TMP_AUTH_GCS_ACCESS_PATH, "w") as outfile:
                dump(self._keyfile_dict, outfile)
            environ[
                "GOOGLE_APPLICATION_CREDENTIALS"
            ] = settings.TMP_AUTH_GCS_ACCESS_PATH

    def parse_connector_url(self, url: str):
        """Parses and validates a google cloud storage url.

        Creates the Filesystem connection.
        Returns:
            tuple(bucket_name, blob).
        """
        try:
            spec = parse_gcs_path(url)
            self.bucket_name = spec.bucket
            return spec.bucket, spec.blob
        except GCSPathError as exc:
            raise DataConnectorsException from exc

    def get_blob(self, blob, bucket_name=None):
        """Get a file in Google Cloud Storage.

        Args:
            blob: `str`. the path to the object to check in the Google cloud storage bucket.
            bucket_name: `str`. Name of the bucket in which the file is stored
        """
        if not bucket_name:
            bucket_name, blob = self.parse_connector_url(blob)

        bucket = self.get_bucket(bucket_name)
        # Wrap google.cloud.storage's blob to raise if the file doesn't exist
        obj = bucket.get_blob(blob)

        if obj is None:
            raise DataConnectorsException(
                "File does not exist: {}".format(blob))

        return obj

    def get_bucket(self, bucket_name):
        """Gets a bucket by name.

        Args:
            bucket_name: `str`. Name of the bucket
        """
        return self.client.get_bucket(bucket_name)

    def check_blob(self, blob_name: str, bucket_name: Optional[str] = None):
        """Checks for the existence of a file in Google Cloud Storage.

        Args:
            blob_name: `str`. the path to the object to check in the Google cloud storage bucket.
            bucket_name: `str`. Optional. Name of the bucket in which the file is stored
        """
        try:
            return bool(self.get_blob(blob=blob_name, bucket_name=bucket_name))
        except Exception as exc:
            logger.logger.info("Block does not exist %s", exc)
            return False

    def delete_blob_if_exists(self, blob_name: str, bucket_name: Optional[str] = None):
        "Deletes a BLOB from GCS if exists"
        if bucket_name is None:
            bucket_name, blob_name = self.parse_connector_url(blob_name)

        bucket = self.get_bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=blob_name)

        for blob in blobs:  # If no blobs exist, just skips
            blob.delete()

    def ls(self, path: str):
        "Lists files and dirs."
        results = self.list(key=path)
        return {"files": results["blobs"], "dirs": results["prefixes"]}

    def list(
        self, key, bucket_name=None, path=None, delimiter="/", blobs=True, prefixes=True
    ):
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
        if not bucket_name:
            bucket_name, key = self.parse_connector_url(key)

        bucket = self.get_bucket(bucket_name)

        if key and not key.endswith("/"):
            key += "/"

        prefix = key
        if path:
            prefix = os_path.join(prefix, path)

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        def get_iterator():
            return bucket.list_blobs(prefix=prefix, delimiter=delimiter)

        def get_blobs(_blobs):
            list_blobs = []
            for blob in _blobs:
                name = blob.name[len(key):]
                size = blob.size
                if all([name, size]):
                    list_blobs.append((name, blob.size))
            return list_blobs

        def get_prefixes(_prefixes):
            list_prefixes = []
            for folder_path in _prefixes:
                name = folder_path[len(key): -1]
                list_prefixes.append(name)
            return list_prefixes

        results = {"blobs": [], "prefixes": []}

        if blobs:
            iterator = get_iterator()
            results["blobs"] = get_blobs(list(iterator))

        if prefixes:
            iterator = get_iterator()
            for page in iterator.pages:
                results["prefixes"] += get_prefixes(page.prefixes)

        return results

    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        "Given a path, return the valid files for a given file_type."
        if extension is None:
            files = self.ls(path)["files"]
            file_paths = [
                os_path.join(path, file[0])
                for file in files
                if self.check_file_extension(file[0]) == file_type.value
            ]
        else:
            file_paths = [path]
        return file_paths

    def test(self):
        _ = self.client.list_buckets(1)
