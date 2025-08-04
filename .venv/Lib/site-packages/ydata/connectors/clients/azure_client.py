from __future__ import absolute_import, division, print_function

from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

from ydata.connectors.utils import get_from_env


def get_account_key(keys=None):
    keys = keys or ["AZURE_ACCOUNT_KEY"]
    return get_from_env(keys)


def get_connection_string(keys=None):
    keys = keys or ["AZURE_CONNECTION_STRING"]
    return get_from_env(keys)


def get_blob_service_connection(connection_string=None):
    connection_string = connection_string or get_connection_string()
    return BlobServiceClient.from_connection_string(connection_string)


def get_datalake_service_connection(connection_string=None):
    connection_string = connection_string or get_connection_string()
    return DataLakeServiceClient.from_connection_string(connection_string)


def parse_path(path):
    path_split = path.split("/")
    url_index = [
        path_split.index(val)
        for val in path_split
        if val.endswith("blob.core.windows.net")
    ][0]
    account = path_split[url_index].split(".")[0]
    container = path_split[url_index + 1]
    try:
        filename = path_split[url_index + 2]
    except Exception:
        filename = None
    return account, container, filename
