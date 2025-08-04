from _typeshed import Incomplete
from enum import Enum
from ydata.connectors.base_connector import BaseConnector
from ydata.core.connectors import WriteMode
from ydata.dataset import Dataset

class Cloud(Enum):
    AZURE = ...
    AWS = ...
    @property
    def connector(self): ...

def manage_exceptions(func): ...

class DatabricksLakehouse(BaseConnector):
    """Add here some more details."""
    DB_TYPE: str
    DB_PYTHON_LIBRARY: str
    uri: Incomplete
    catalog: Incomplete
    schema: Incomplete
    def __init__(self, host: str, access_token: str, staging_credentials: dict, cloud: Cloud | str = ..., catalog: str | None = None, schema: str | None = None, echo: bool = False) -> None: ...
    @property
    def client(self):
        """Getter method for client property. Lazy-eval: create only if does not exist."""
    def set_client(self) -> None:
        """
        Sets a new databricks cluster client
        Returns
        -------
        """
    @property
    def connection(self):
        """Returns the connection engine."""
    def list_sqlwarehouses(self) -> dict:
        """
        List with the names and properties to connect to the available SQL warehouse engines
        Returns a list with the available SQL Warehouses for the computation
        -------
        """
    def connect_to_warehouse(self, warehouse: dict):
        """
        Method to set the URI and connect with the requested warehouse
        Parameters
        ----------
        warehouse - a dictionary with the properties of the warehouse such as hostname and path.

        Returns A valid cursor for querying and writing.
        -------
        """
    def list_catalogs(self) -> list:
        """
            Get the list of available catalogs in the current Databricks lakehouse
        Returns a list with the catalogs names
        ------
        """
    def list_schemas(self, catalog: str) -> list:
        """
            Get the list of available schemas given a catalog in Databricks lakehouse cluster context
        Returns a list with the schemas name for a given catalog
        -------
        """
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
    def get_table(self, table: str, warehouse: str, catalog: str | None = None, schema: str | None = None) -> Dataset:
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
    def get_table_sample(self, table: str, warehouse: str, sample_size: int = ..., catalog: str | None = None, schema: str | None = None):
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
    def query_sample(self, query: str, warehouse: str, sample_size: int = ...) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by adding a suffix to
        the provided query."""
    def write_table(self, data: Dataset, staging_path: str, warehouse: str, table: str, catalog: str | None = None, schema: str | None = None, if_exists: str | WriteMode = ...):
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
    def test(self) -> None: ...
