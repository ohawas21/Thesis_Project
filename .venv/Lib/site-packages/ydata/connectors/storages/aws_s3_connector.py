"""Implementation of a Connector that allows to read/write data and list
contents.

from Amazon S3.
"""
from __future__ import absolute_import, division, print_function

from collections import namedtuple
from os import environ
from os import path as os_path
from urllib import parse

from botocore.exceptions import ClientError

from ydata.connectors.clients import aws_client
from ydata.connectors.exceptions import DataConnectorsException, S3PathError
from ydata.dataset.filetype import FileType
from ydata.connectors.logger import logger
from ydata.connectors.storages import _S3_STORAGE
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector


class S3Spec(namedtuple("S3Spec", "bucket key")):
    """A specification for s3 configuration."""


def parse_s3_path(s3_path):
    """Parses and validates an S3 url.

    Returns:
         tuple(bucket_name, key).
    """
    parsed_url = parse.urlparse(s3_path)
    if not parsed_url.netloc:
        raise S3PathError('Received an invalid S3 url `{}`'.format(s3_path))
    else:
        bucket_name = parsed_url.netloc
        key = parsed_url.path.strip('/')
        return S3Spec(bucket_name, key)


class S3Connector(ObjectStorageConnector):
    """S3 store service using Modian."""

    STORAGE_TYPE = _S3_STORAGE

    def __init__(
        self,
        access_key_id,
        secret_access_key,
        aws_ssl=True,
        session_token=None,
        aws_region=None,
        ds=None,
        **kwargs
    ):
        super().__init__()
        self._resource = None

        if ds is None:
            self._endpoint_url = (
                kwargs.get("endpoint_url")
                or kwargs.get("aws_endpoint_url")
                or kwargs.get("AWS_ENDPOINT_URL")
            )

            self._aws_access_key_id = access_key_id
            self._aws_secret_access_key = secret_access_key
            self._aws_session_token = session_token
            self._region_name = aws_region
            self._aws_use_ssl = aws_ssl

            # Not mandatory anymore for that reason will be kept as a kwarg
            self._aws_verify_ssl = kwargs.get(
                "verify_ssl",
                kwargs.get("aws_verify_ssl", kwargs.get(
                    "AWS_VERIFY_SSL", None)),
            )

            self.credentials = {
                "aws_access_key_id": self._aws_access_key_id,
                "aws_secret_access_key": self._aws_secret_access_key,
                "aws_session_token": self._aws_session_token,
                "aws_verify_ssl": self._aws_verify_ssl,
                "aws_use_ssl": self._aws_use_ssl,
                "endpoint_url": self._endpoint_url,
                "region_name": self._region_name,
            }
        else:
            self.credentials = ds.credentials

        self._filesystem = self.filesystem
        self.storage_options = {
            "key": self._aws_access_key_id,
            "secret": self._aws_secret_access_key,
        }

    def set_client(self):
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
        self._client = aws_client.get_aws_client("s3", **self.credentials)

    def set_env_vars(self):
        if self.credentials["endpoint_url"]:
            environ["AWS_ENDPOINT_URL"] = self.credentials["endpoint_url"]
        if self.credentials["aws_access_key_id"]:
            environ["AWS_ACCESS_KEY_ID"] = self.credentials["aws_access_key_id"]
        if self.credentials["aws_secret_access_key"]:
            environ["AWS_SECRET_ACCESS_KEY"] = self.credentials["aws_secret_access_key"]
        if self.credentials["aws_session_token"]:
            environ["AWS_SECURITY_TOKEN"] = self.credentials["aws_session_token"]
        if self.credentials["region_name"]:
            environ["AWS_REGION"] = self.credentials["region_name"]
        if self.credentials["aws_use_ssl"] is not None:
            environ["AWS_USE_SSL"] = str(self.credentials["aws_use_ssl"])
        if self.credentials["aws_verify_ssl"] is not None:
            environ["AWS_VERIFY_SSL"] = self.credentials["aws_verify_ssl"]

    @property
    def filesystem(self):
        "Gets or sets the S3 Filesystem based on the provided credentials."
        # define the aws client using the giving credentials and s3fs
        self.set_filesystem(self.credentials)
        return self._filesystem

    def set_filesystem(self, credentials):
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
        self._filesystem = aws_client.get_aws_filesystem(credentials)

    @property
    def resource(self):
        "Gets or sets a Boto3 Resource Service based on the provided credentials."
        if self._resource is None:
            kwargs = self.credentials
            self.set_resource(**kwargs)
        return self._resource

    def set_resource(self, **kwargs):
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
        self._resource = aws_client.get_aws_resource(
            "s3",
            endpoint_url=kwargs.get("endpoint_url"),
            aws_access_key_id=kwargs.get("aws_access_key_id"),
            aws_secret_access_key=kwargs.get("aws_secret_access_key"),
            aws_session_token=kwargs.get("aws_session_token"),
            region_name=kwargs.get("region"),
        )

    def parse_connector_url(self, url: str):
        """Parses and validates an S3 url.

        Returns:
             tuple(bucket_name, key).
        """
        try:
            spec = parse_s3_path(url)
            return spec.bucket, spec.key
        except Exception as exc:
            raise DataConnectorsException(
                "Please provide a valid S3 url.") from exc

    @staticmethod
    def check_prefix_format(prefix: str, delimiter: str):
        if not delimiter or not prefix:
            return prefix
        return prefix + delimiter if prefix[-1] != delimiter else prefix

    def check_bucket(self, bucket_name):
        """Checks if a bucket exists.

        Args:
            bucket_name: `str`. Name of the bucket
        """
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            logger.info(e.response["Error"]["Message"])
            return False

    def get_bucket(self, bucket_name):
        """Gets a bucket by name.

        Args:
            bucket_name: `str`. Name of the bucket
        """
        return self.resource.Bucket(bucket_name)

    def get_key(self, key, bucket_name=None):
        """
        Returns a boto3.s3.Object
        Args:
            key: `str`. the path to the key.
            bucket_name: `str`. the name of the bucket.
        """
        if not bucket_name:
            (bucket_name, key) = self.parse_connector_url(key)

        try:
            obj = self.resource.Object(bucket_name, key)
            obj.load()
            return obj
        except Exception as e:
            raise DataConnectorsException(e)

    def read_key(self, key, bucket_name=None):
        """
        Reads a key from S3
        Args:
            key: `str`. S3 key that will point to the file.
            bucket_name: `str`. Name of the bucket in which the file is stored.
        """

        obj = self.get_key(key, bucket_name)
        return obj.get()["Body"].read().decode("utf-8")

    def ls(self, path):
        (bucket_name, key) = self.parse_connector_url(path)
        results = self.list(bucket_name=bucket_name, prefix=key)
        return {"files": results["keys"], "dirs": results["prefixes"]}

    def list(
        self,
        bucket_name,
        prefix="",
        delimiter="/",
        page_size=None,
        max_items=None,
        keys=True,
        prefixes=True,
    ):
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
        config = {
            "PageSize": page_size,
            "MaxItems": max_items,
        }

        paginator = self.client.get_paginator("list_objects_v2")

        prefix = self.check_prefix_format(prefix=prefix, delimiter=delimiter)
        response = paginator.paginate(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter=delimiter,
            PaginationConfig=config,
        )

        def get_keys(contents):
            list_keys = []
            for cont in contents:
                if cont.get("Size") > 0:  # avoid listing empty contents
                    list_keys.append(
                        (cont["Key"][len(prefix):], cont.get("Size")))
            return list_keys

        def get_prefixes(page_prefixes):
            list_prefixes = []
            for pref in page_prefixes:
                list_prefixes.append(pref["Prefix"][len(prefix): -1])
            return list_prefixes

        results = {"keys": [], "prefixes": []}
        for page in response:
            if prefixes:
                results["prefixes"] += get_prefixes(
                    page.get("CommonPrefixes", []))
            if keys:
                results["keys"] += get_keys(page.get("Contents", []))

        return results

    def list_keys(
        self, bucket_name, prefix="", delimiter="", page_size=None, max_items=None
    ):
        """
        Lists keys in a bucket under prefix and not containing delimiter
        Args:
            bucket_name: `str`. the name of the bucket
            prefix: `str`. a key prefix
            delimiter: `str`. the delimiter marks key hierarchy.
            page_size: `int`. pagination size
            max_items: `int`. maximum items to return
        """
        results = self.list(
            bucket_name=bucket_name,
            prefix=prefix,
            delimiter=delimiter,
            page_size=page_size,
            max_items=max_items,
            keys=True,
            prefixes=False,
        )
        return results["keys"]

    def check_key(self, key, bucket_name=None):
        """
        Checks if a key exists in a bucket
        Args:
            key: `str`. S3 key that will point to the file
            bucket_name: `str`. Name of the bucket in which the file is stored
        """
        if not bucket_name:
            (bucket_name, key) = self.parse_connector_url(key)

        try:
            self.client.head_object(Bucket=bucket_name, Key=key)
            return True
        except ClientError as exc:
            logger.info(exc.response["Error"]["Message"])
            return False

    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        if extension is None:
            files = self.ls(path)["files"]
            file_paths = [
                os_path.join(path, file[0]).replace("S3", "s3")
                for file in files
                if self.check_file_extension(file[0]) == file_type.value
            ]
        else:
            file_paths = [path.replace("S3", "s3")]

        return file_paths

    def test(self):
        _ = self.client.list_buckets()
