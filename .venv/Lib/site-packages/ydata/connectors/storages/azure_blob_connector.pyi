from _typeshed import Incomplete
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.dataset.filetype import FileType as FileType

class AzureBlobConnector(ObjectStorageConnector):
    """Azure Blob storage connector."""
    STORAGE_TYPE: Incomplete
    credentials: Incomplete
    storage_options: Incomplete
    def __init__(self, account_name: str, account_key: str) -> None:
        """
        Inherits from ObjectStorageConnector and gets Azure Storage Account credentials based
        on the connection_string
        Args:
            - connection_string: 'str'
        """
    @property
    def client(self):
        """Creates a client connection to ABFS."""
    def set_client(self, connection_string: Incomplete | None = None) -> None:
        """Sets up a new Azure storage account connection.
        Args:
            -connection_string: 'str' If a different connection_string is provided, overrides current connection
        """
    @staticmethod
    def parse_connector_url(url: str):
        """
        Parses a URL to an Azure Blob Storage object
        Args:
            url: 'str' A valid URL link to an object inside the current Azure Blob Storage account                Must conform either one of the following schemas:
                    abfs schema: abfs://(storage_acc/)?(.blob.core.windows.net/)?container_name/blob_name
                    https schema: https://(storage_acc)?(.blob.core.windows.net/)?container_name/blob_name
        Returns:
            container_name: 'str' Name of the container
            blob_name: 'str' Name of the blob (can be None)
        """
    def list(self, path: Incomplete | None = None, container_name_starts_with: Incomplete | None = None, file_type: Incomplete | None = None, include_metadata: bool = False):
        """Returns a list of contents structured in a dictionary with keys.

        ['blobs', 'containers'] Supports the following modes of operation:

        1) path = None: Lists all containers found in the storage account
            2) path = path: Lists all blobs found in the implied container and starting with implied prefix. If no prefix is inferred lists all contents of the implied container
        When not passing a path, the argument container_name_starts_with can be defined to filter available containers with a string prefix.
        Also when not passing a path, the include_metadata flag can be set to True to retrieve container metadata in the container results.
        When using list to retrieve blobs, file_type can be passed to filter returned blobs by file_type.
        Will return for blob results a list with string 'CONTAINER_NOT_EXISTING' if the container cannot be found.
        """
    def check_blob(self, path: str) -> bool:
        """Checks for the existence of a blob in Azure Blob Storage.

        Args:
            path: `str`. the path to the object to check in the Azure Blob Storage storage.
        """
    def check_container(self, container_name: str) -> bool:
        """Checks for the existence of a container in Azure Blob Storage.

        Args:
            container_name: `str`. the name of the container to check for in the Azure Blob Storage storage.
        """
    def get_blob(self, url: str = None):
        """Composes a list of elligible blob clients based on a passed url.

        Examples of operation based on url:
            1) abfs://container_name/blob_name -> returns blob_name located in container_name
            2) abfs://container_name/* -> returns all blobs located in container_name
            3) abfs://container_name/*.extension ->  returns all blobs located in container_name with the specified extension
        Returns:
            A list of BlobClients
        """
    def get_container(self, container_name):
        """Gets a container by name.

        Args:
            container_name: 'str'. Name of the container
        Returns:
            'ContainerClient' A client that interacts with the requested container
        """
    def delete_blob_if_exists(self, blob_name: str, container_name: str | None = None):
        """
        Deletes a blob from the Azure Blob Storage if it exists
        If no container_name is passed, will assume blob_name to be a connector url
        Args:
            blob_name: 'str'. Name of the blob or blob connector url
            container_name: 'str'. Name of the container
        """
    def get_file_paths(self, path: str, file_type: FileType, extension: str):
        """Given a path, return the valid files for a given file_type."""
    def write_file(self, data, path: str, file_type: FileType | None = None, *args, **kwargs): ...
    def test(self) -> None: ...
