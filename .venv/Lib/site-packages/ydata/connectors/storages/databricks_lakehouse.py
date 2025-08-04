"""Python class to connect to Databricks Lakehouse."""
from enum import Enum

from dask import dataframe as dddataframe
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
from pandas import read_sql_query
from sqlalchemy import create_engine, text

from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.exceptions import (InvalidLakehouseTokenException, InvalidQuery, LakehouseConnectorException,
                                         NotFoundLakehouseException)
from ydata.connectors.storages import _MAX_SAMPLE
from ydata.connectors.storages.aws_s3_connector import S3Connector
from ydata.connectors.storages.azure_blob_connector import AzureBlobConnector
from ydata.core.connectors import WriteMode
from ydata.dataset import Dataset


class Cloud(Enum):
    AZURE = 'azure', AzureBlobConnector
    AWS = 'aws', S3Connector

    @classmethod
    def _missing_(cls, value):

        value = value.lower()
        for member in cls:
            if member.value[0].lower() == value:
                return member
        return None

    @property
    def connector(self):
        return self.value[1]


def manage_exceptions(func):
    def func_exceptions(self, *arg, **kwargs):

        try:
            return func(self, *arg, **kwargs)
        except NotFound as exc:
            if exc.error_code == 'CATALOG_DOES_NOT_EXIST':
                raise NotFoundLakehouseException("The provided Catalog name - {} - does not exist in the current Databricks workspace."
                                                 "Please validate you input.".format(kwargs['catalog']))
            elif exc.error_code == 'SCHEMA_DOES_NOT_EXIST':
                raise NotFoundLakehouseException("The provided Schema name - {} - does not exist in the current Databricks workspace."
                                                 "Please validate you input.".format(kwargs['schema']))
            else:
                raise NotFoundLakehouseException("The provided workspace {} cannot be found. Please validate the provided input."
                                                 "Make sure to match the following format https://instance_name.cloud.databricks.com/".format(self._host))
        except PermissionDenied:
            raise InvalidLakehouseTokenException("The token provided does not have permissions to access this workspace. "
                                                 "Please validate your access token in your Databricks account.")

    return func_exceptions


class DatabricksLakehouse(BaseConnector):
    """Add here some more details."""
    DB_TYPE = "lakehouse"
    DB_PYTHON_LIBRARY = "databricks"
    _COPY_INTO = """COPY INTO {}.{}.{} FROM '{}' FILEFORMAT = PARQUET COPY_OPTIONS ('mergeSchema' = 'true');"""
    _SAMPLE_SUFFIX = " LIMIT {};"
    _URI_FORMAT = "{}://token:{}@{}?http_path={}"

    _client = None
    _conn = None
    uri = None

    def __init__(self,
                 host: str,
                 access_token: str,
                 staging_credentials: dict,
                 cloud: Cloud | str = Cloud.AWS,
                 catalog: str | None = None,
                 schema: str | None = None,
                 echo: bool = False):
        super().__init__()

        self._access_token = access_token
        self._host = host

        # init the properties for the staging connector
        self._staging_connector = Cloud(cloud).connector(**staging_credentials)

        self.catalog = catalog
        self.schema = schema

        self._echo = echo

    @property
    def client(self):
        "Getter method for client property. Lazy-eval: create only if does not exist."
        if self._client is None:
            self.set_client()
        return self._client

    def set_client(self):
        """
        Sets a new databricks cluster client
        Returns
        -------
        """
        self._client = WorkspaceClient(
            host=self._host, token=self._access_token)

    @property
    def connection(self):
        """Returns the connection engine."""
        if self._conn is None and self.uri:
            self._conn = create_engine(self.uri,
                                       connect_args={
                                           "_user_agent_entry": 'ydatafabric/2.35'},
                                       echo=self._echo)
        return self._conn

    @manage_exceptions
    def list_sqlwarehouses(self) -> dict:
        """
        List with the names and properties to connect to the available SQL warehouse engines
        Returns a list with the available SQL Warehouses for the computation
        -------
        """
        warehouses = {warehouse.name: {'hostname': warehouse.odbc_params.hostname,
                                       'path': warehouse.odbc_params.path,
                                       'port': warehouse.odbc_params.port,
                                       'protocol': warehouse.odbc_params.protocol} for warehouse in self.client.warehouses.list()}
        return warehouses

    def connect_to_warehouse(self, warehouse: dict):
        """
        Method to set the URI and connect with the requested warehouse
        Parameters
        ----------
        warehouse - a dictionary with the properties of the warehouse such as hostname and path.

        Returns A valid cursor for querying and writing.
        -------
        """
        try:
            self.uri = self._URI_FORMAT.format(self.DB_PYTHON_LIBRARY,
                                               self._access_token,
                                               warehouse['hostname'],
                                               warehouse['path'])

            cursor = self.connection.connect()

        except Exception as exc:
            raise Exception(
                "Something went wrong while trying to connect with the provided warehouse: {} .".format(exc))

        return cursor

    @manage_exceptions
    def list_catalogs(self) -> list:
        """
            Get the list of available catalogs in the current Databricks lakehouse
        Returns a list with the catalogs names
        ------
        """
        return [catalog.name for catalog in self.client.catalogs.list()]

    @manage_exceptions
    def list_schemas(self, catalog: str) -> list:
        """
            Get the list of available schemas given a catalog in Databricks lakehouse cluster context
        Returns a list with the schemas name for a given catalog
        -------
        """
        return [schema.name for schema in self.client.schemas.list(catalog)]

    @manage_exceptions
    def list_tables(self, catalog: str, schema: str) -> list:
        """
            Get the list of tables available for a schema in a Catalog
        Parameters
        ----------
        catalog Name of the catalog
        schema Name of the table that belong to the catalog
        Returns the list of tables available
        -------
        """

        return [table.name for table in self.client.tables.list(catalog, schema)]

    def __validate_warehouse(self, warehouse: str):
        try:
            warehouse = self.list_sqlwarehouses()[warehouse]
        except Exception:
            raise LakehouseConnectorException('Invalid warehouse with the name {}.'
                                              'Please input a different warehouse name'.format(warehouse))

        return warehouse

    def __set_inputs(self, catalog: str, schema: str) -> tuple:
        """Helper function to set the catalog and schema inputs depending on
        how the Connector was init."""
        if catalog is None and self.catalog is None:
            raise LakehouseConnectorException(
                "Please provide a Catalog as input.")

        if schema is None and self.schema is None:
            raise LakehouseConnectorException(
                "Please provide a Schema as input.")

        if catalog is None:
            catalog = self.catalog

        if schema is None:
            schema = self.schema

        return catalog, schema

    def get_table(self, table: str,
                  warehouse: str,
                  catalog: str | None = None,
                  schema: str | None = None) -> Dataset:
        """
        Method to read a full table from Databricks delta lake
        Parameters
        ----------
        catalog The Catalog that the table belongs to
        schema The schema to which the table belongs to
        table The name of the table that you want to read

        Returns   A Dataset object with the data from the requested table
        -------
        """
        catalog, schema = self.__set_inputs(catalog, schema)

        query = 'SELECT * ' \
                'FROM {}.{}.{}'.format(catalog, schema, table)

        return self.query(query=query,
                          warehouse=warehouse)

    def get_table_sample(self, table: str,
                         warehouse: str,
                         sample_size: int = _MAX_SAMPLE,
                         catalog: str | None = None,
                         schema: str | None = None):
        """
        Method to read a sample fo a table from Databricks Delta Lake
        Parameters
        ----------
        table the name of the table
        warehouse the name of the warehouse to be used to query the table
        catalog the name of the catalog that the table belongs to
        schema the name of the schema that the table belongs to
        sample_size the number of rows to read from the table

        Returns A Dataset object with the data from the requested table
        -------
        """
        catalog, schema = self.__set_inputs(catalog, schema)

        query = 'SELECT * ' \
                'FROM {}.{}.{}'.format(catalog, schema, table)

        return self.query_sample(query=query,
                                 warehouse=warehouse,
                                 sample_size=sample_size)

    def _query(self, query: str) -> Dataset:
        try:
            iter_data = read_sql_query(
                sql=query, con=self.connection, chunksize=50000)

            data = dddataframe.multi.concat(
                [dddataframe.from_pandas(chunk, npartitions=3)
                 for chunk in iter_data]
            )
        except Exception as exc:
            raise InvalidQuery(f"Invalid query provide: {exc}") from exc

        return Dataset(data)

    def query(self, query: str, warehouse: str) -> Dataset:
        """
        Method that returns the data from a query made against Databricks delta lake tables
        Parameters
        ----------
        query A string expression compatible with Databricks SQL syntax
        warehouse The SQL Warehouse to be used for the query

        Returns A dataset object with the data from the query
        -------
        """
        warehouse = self.__validate_warehouse(warehouse)

        cursor = self.connect_to_warehouse(warehouse)

        data = self._query(query)

        cursor.close()

        return data

    def query_sample(
        self,
        query: str,
        warehouse: str,
        sample_size: int = _MAX_SAMPLE,
    ) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by adding a suffix to
        the provided query."""
        sample_suffix = self._SAMPLE_SUFFIX.format(sample_size)

        if query.endswith(";"):
            query = query.replace(";", "", 1) + sample_suffix
        else:
            query += sample_suffix
        return self.query(query=query, warehouse=warehouse)

    def write_table(self,
                    data: Dataset,
                    staging_path: str,
                    warehouse: str,
                    table: str,
                    catalog: str | None = None,
                    schema: str | None = None,
                    if_exists: str | WriteMode = WriteMode.FAIL):
        """Write a new table to Databricks Lakehouse. The created data will
        automatically be made available in Databricks Unity Catalog.

        Parameters
        ----------
        staging_path The path of an AWS S3 or Azure Storage to stage the data
        catalog The name of the catalog where the data should be created
        schema The name of the schema where the data should be created
        table The name of the table
        if_exists Defines  the strategy in case the table already exists - replace, append or fail
        """
        if_exists = WriteMode(if_exists)
        catalog, schema = self.__set_inputs(catalog, schema)

        warehouse = self.__validate_warehouse(warehouse)
        cursor = self.connect_to_warehouse(warehouse)

        # Writing the data to the staging area
        self._staging_connector.write_file(data=data,
                                           path=staging_path,
                                           file_type='parquet',
                                           if_exists=WriteMode.REPLACE)

        try:
            if if_exists == WriteMode.FAIL:
                if table in self.list_tables(catalog=catalog, schema=schema):
                    raise Exception(
                        "The table {} already exists. Please provide a new table name.".format(table))
            elif WriteMode.REPLACE:
                cursor.execute(text('DROP TABLE IF EXISTS {}.{}.{};'.format(catalog,
                                                                            schema,
                                                                            table)))

            cursor.execute(text('CREATE TABLE IF NOT EXISTS {}.{}.{};'.format(catalog,
                                                                              schema,
                                                                              table)))

            cursor.execute(text(self._COPY_INTO.format(catalog,
                                                       schema,
                                                       table,
                                                       staging_path)))
            cursor.close()
        except Exception as exc:
            raise Exception(
                "Something went wrong while writing to Delta Lake: {} .".format(exc))

    def test(self):
        _ = self.list_catalogs()
        self._staging_connector.test()
