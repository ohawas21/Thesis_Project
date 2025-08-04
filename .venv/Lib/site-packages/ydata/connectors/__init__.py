"""The :mod:`ydata.connectors` module that gathers the data connectors that
will be made available at YData's platform."""
from ydata.connectors.catalogs.databricks_unity import DatabricksUnityCatalog
from ydata.connectors.storages.aws_s3_connector import S3Connector
from ydata.connectors.storages.azure_blob_connector import AzureBlobConnector
from ydata.connectors.storages.big_query_connector import BigQueryConnector
from ydata.connectors.storages.databricks_lakehouse import DatabricksLakehouse
from ydata.connectors.storages.gcs_connector import GCSConnector
from ydata.connectors.storages.local_connector import LocalConnector
from ydata.connectors.storages.rdbms_connector import (AzureSQLConnector, MySQLConnector, PostgreSQLConnector,
                                                       SnowflakeConnector)

__all__ = [
    "S3Connector",
    "AzureBlobConnector",
    "GCSConnector",
    "MySQLConnector",
    "AzureSQLConnector",
    "BigQueryConnector",
    "LocalConnector",
    "PostgreSQLConnector",
    "SnowflakeConnector",
    "DatabricksUnityCatalog",
    "DatabricksLakehouse"
]
