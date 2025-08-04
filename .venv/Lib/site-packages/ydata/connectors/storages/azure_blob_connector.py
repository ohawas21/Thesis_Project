from __future__ import absolute_import, division, print_function

import os
import re
from typing import Optional

from azure.storage.blob import BlobClient, ContainerClient

from ydata.connectors.clients import azure_client
from ydata.connectors.exceptions import DataConnectorsException
from ydata.dataset.filetype import FileType
from ydata.connectors.storages import _AZURE_STORAGE
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector


class AzureBlobConnector(ObjectStorageConnector):
    """Azure Blob storage connector."""

    STORAGE_TYPE = _AZURE_STORAGE

    def __init__(self, account_name: str, account_key: str):
        """
        Inherits from ObjectStorageConnector and gets Azure Storage Account credentials based
        on the connection_string
        Args:
            - connection_string: 'str'
        """
        super().__init__()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
        self.credentials = {"connection_string": connection_string}

        self.storage_options = {
            "account_name": account_name,
            "account_key": account_key,
            "connection_string": connection_string,
        }

        self._client = self.client

    @property
    def client(self):
        """Creates a client connection to ABFS."""
        self.set_client(self.storage_options["connection_string"])
        return self._client

    def set_client(self, connection_string=None):
        """Sets up a new Azure storage account connection.
        Args:
            -connection_string: 'str' If a different connection_string is provided, overrides current connection
        """
        if not connection_string:
            connection_string = self.connection_string
        self._client = azure_client.get_blob_service_connection(
            connection_string)

    def _get_storage_account_information(self):
        """Retrieves Azure Storage Account information from the client.

        Returns:
            account_info: 'dict'
        """
        account_info = self.client.get_account_information()
        return account_info

    @staticmethod
    def parse_connector_url(url: str):
        """
        Parses a URL to an Azure Blob Storage object
        Args:
            url: 'str' A valid URL link to an object inside the current Azure Blob Storage account\
                Must conform either one of the following schemas:
                    abfs schema: abfs://(storage_acc/)?(.blob.core.windows.net/)?container_name/blob_name
                    https schema: https://(storage_acc)?(.blob.core.windows.net/)?container_name/blob_name
        Returns:
            container_name: 'str' Name of the container
            blob_name: 'str' Name of the blob (can be None)
        """
        if ".dfs." in url:
            regex = r"(?P<valid_scheme>^abfss://)(?P<container>[a-zA-Z0-9-]{3,24})@(?P<account>[a-zA-Z0-9]{3,24})(?P<abfs_domain>.dfs.core.windows.net)(/|$)?(.*/(?P<blob>.+)$)"
        else:
            regex = r"(?P<valid_scheme>^(abfs|https)://)(?P<account>[a-zA-Z0-9]{3,24}@[a-zA-Z\-]+\.[a-zA-Z]+)?(?P<abfs_domain>.blob.core.windows.net)?/?(?P<container>[\w\.-]+)(/|$)(?P<blob>([\*\w\.-]+/?)*[\*\w\.-]+|$)"
        scheme_pat = re.compile(regex)
        match = re.match(scheme_pat, url)
        if not match:
            raise DataConnectorsException(
                "The provided URL does not conform to the correct schema"
            )
        return match.group("container"), match.group("blob")

    def _list_containers(self, name_starts_with=None, include_metadata=False):
        """Returns the list of the existing containers for a given account.

        Optionally with container metadata.
        Args:
            startswith: 'str'. Beginning of the container name that we are looking for
        Returns: A list of tuples with format (container.name, container.metadata).
        The metadata field of the tuples will be None when include_metadata is set to False.
        """
        all_containers = []
        for container in self.client.list_containers(
            name_starts_with=name_starts_with, include_metadata=include_metadata
        ):
            all_containers.append((container.name, container.metadata))
        return all_containers

    def _list_blobs(self, container_name, name_starts_with=None, file_type=None):
        """Gets the list of existing blobs in a container.

        Args:
            container_name: 'str'. Name of the container
        Returns:
            'list'. A list with all existing blobs
        """
        if file_type:
            file_type_ = self.parse_file_type(file_type)

        all_blobs = []
        container_client = self.client.get_container_client(
            container=container_name)
        if self._object_exists(container_client):
            for blob in container_client.list_blobs(name_starts_with=name_starts_with):
                if file_type:
                    if not blob.name.endswith(
                        file_type_.value
                    ):  # file_type does not match
                        continue  # Go to next blob
                all_blobs.append((blob.name, blob.size))
        else:
            all_blobs.append("CONTAINER_NOT_EXISTING")
        return all_blobs

    def list(
        self,
        path=None,
        container_name_starts_with=None,
        file_type=None,
        include_metadata=False,
    ):
        """Returns a list of contents structured in a dictionary with keys.

        ['blobs', 'containers'] Supports the following modes of operation:

        1) path = None: Lists all containers found in the storage account
            2) path = path: Lists all blobs found in the implied container and starting with implied prefix. If no prefix is inferred lists all contents of the implied container
        When not passing a path, the argument container_name_starts_with can be defined to filter available containers with a string prefix.
        Also when not passing a path, the include_metadata flag can be set to True to retrieve container metadata in the container results.
        When using list to retrieve blobs, file_type can be passed to filter returned blobs by file_type.
        Will return for blob results a list with string 'CONTAINER_NOT_EXISTING' if the container cannot be found.
        """
        results = {"containers": [], "blobs": []}
        if not path:
            results["containers"] = self._list_containers(
                name_starts_with=container_name_starts_with,
                include_metadata=include_metadata,
            )
        else:  # key assumed to be a path
            container_name, prefix = self.parse_connector_url(path)
            results["blobs"] = self._list_blobs(
                container_name, name_starts_with=prefix, file_type=file_type
            )
        return results

    def check_blob(self, path: str) -> bool:
        """Checks for the existence of a blob in Azure Blob Storage.

        Args:
            path: `str`. the path to the object to check in the Azure Blob Storage storage.
        """
        blob_client = self.get_blob(url=path)[0]
        return self._object_exists(blob_client)

    def check_container(self, container_name: str) -> bool:
        """Checks for the existence of a container in Azure Blob Storage.

        Args:
            container_name: `str`. the name of the container to check for in the Azure Blob Storage storage.
        """
        container_client = self.get_container(container_name=container_name)
        return self._object_exists(container_client)

    def get_blob(self, url: str = None):
        """Composes a list of elligible blob clients based on a passed url.

        Examples of operation based on url:
            1) abfs://container_name/blob_name -> returns blob_name located in container_name
            2) abfs://container_name/* -> returns all blobs located in container_name
            3) abfs://container_name/*.extension ->  returns all blobs located in container_name with the specified extension
        Returns:
            A list of BlobClients
        """
        container_name, blob_name = self.parse_connector_url(url)
        container = self.get_container(container_name)
        if not self._object_exists(container):
            return []  # Non existing container, empty list
        if blob_name.startswith("*"):  # Fuzzy blob name matching
            extension = self.check_file_extension(
                blob_name
            )  # Parse blob name for extension
            if extension:
                return [
                    container.get_blob_client(properties)
                    for properties in container.list_blobs()
                    if properties["name"].endswith(extension)
                ]
            else:
                return [
                    container.get_blob_client(properties)
                    for properties in container.list_blobs()
                ]
        # Single blob matching OR multi-blob matching in dir blobs (parquet files)
        else:
            blob_prop_list = [
                properties
                for properties in container.list_blobs(name_starts_with=blob_name)
            ]
            if len(blob_prop_list) == 1:  # Captured blob
                return [container.get_blob_client(blob_prop_list[0])]
            else:  # Captured nested blobs, like parquet blob dirs
                return [
                    container.get_blob_client(properties)
                    for properties in blob_prop_list
                    if properties.name.split("/")[0] == blob_name
                ]

    def get_container(self, container_name):
        """Gets a container by name.

        Args:
            container_name: 'str'. Name of the container
        Returns:
            'ContainerClient' A client that interacts with the requested container
        """
        return self.client.get_container_client(container_name)

    @staticmethod
    def _object_exists(object_client) -> bool:
        """Despite being referenced in the Azure Blob Storage API for python
        sdk none of the ContainerClient or BlobClient exists method are
        currently implemented.

        This method is a suggested workaround and eventually should be
        deprecated.
        """
        if isinstance(object_client, BlobClient):
            test = object_client.get_blob_properties
        elif isinstance(object_client, ContainerClient):
            test = object_client.get_container_properties
        else:  # Unexpected type of object
            raise TypeError(
                "The method does not support objects of type {}".format(
                    object_client)
            )
        try:
            test()
            return True
        except Exception:
            return False

    def delete_blob_if_exists(
        self, blob_name: str, container_name: Optional[str] = None
    ):
        """
        Deletes a blob from the Azure Blob Storage if it exists
        If no container_name is passed, will assume blob_name to be a connector url
        Args:
            blob_name: 'str'. Name of the blob or blob connector url
            container_name: 'str'. Name of the container
        """
        if container_name is None:
            container_name, blob_name = self.parse_connector_url(blob_name)

        container = self.get_container(container_name)

        blobs = container.list_blobs(name_starts_with=blob_name)

        for blob in blobs:  # If no blobs exist, just skips
            container.delete_blobs(blob)

    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        "Given a path, return the valid files for a given file_type."
        if extension is None:
            files = self.list(path)["blobs"]
            file_paths = [
                os.path.join(path, file[0])
                for file in files
                if self.check_file_extension(file[0]) == file_type.value
            ]
        else:
            file_paths = [path]
        return file_paths

    def write_file(
        self, data, path: str, file_type: Optional[FileType] = None, *args, **kwargs
    ):

        super().write_file(
            data=data,
            path=path,
            file_type=file_type,
            single_file=False,
            *args,
            **kwargs,
        )

    def test(self):
        _ = self._get_storage_account_information()
