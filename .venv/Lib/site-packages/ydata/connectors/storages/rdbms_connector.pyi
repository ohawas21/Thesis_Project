from _typeshed import Incomplete
from dataclasses import dataclass
from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.exceptions import InvalidIndexCol as InvalidIndexCol
from ydata.core.connectors import WriteMode
from ydata.dataset import Dataset
from ydata.dataset.multidataset import MultiDataset
from ydata.dataset.schemas import RDBMSSchema as Schema, Table

@dataclass
class ConnectionString:
    """Connection string settings."""
    username: str
    password: str
    database: str
    schema: str | None = ...
    port: int | None = ...
    hostname: str | None = ...
    warehouse: str | None = ...
    def __init__(self, username, password, database, schema=..., port=..., hostname=..., warehouse=...) -> None: ...

class RDBMSConnector(BaseConnector):
    """RDBMS connectors structure."""
    STORAGE_TYPE: Incomplete
    DB_TYPE: Incomplete
    DB_PYTHON_LIBRARY: Incomplete
    conn_str: Incomplete
    def __init__(self, conn_string: ConnectionString | dict, echo: bool = False) -> None: ...
    @property
    def uri(self): ...
    @property
    def database(self):
        """Returns the name of the selected database to connect to."""
    @property
    def connection(self):
        """Returns the connection engine."""
    def set_schema(self, schema_name: str | None = None):
        """
        Method to set the schema name to be used by the connector
        Parameters
        ----------
        schema_name A string with the name of the database schema
        """
    @property
    def schema_name(self): ...
    def get_database_schema(self, schema_name: str | None = None, refresh: bool = True, set_schema: bool = True) -> Schema:
        """Method to calculate the schema of the selected database to be
        queried."""
    def dispose(self) -> None:
        """Dispose opened connections."""
    def get_tables(self, tables: list[str], schema: Schema | dict | None = None, schema_name: str | None = None):
        """Method to return a v   object from a database This method can return
        the whole database in case tables parameter is not provided."""
    def get_table(self, table: str, schema: Table | dict | None = None, schema_name: str | None = None, index_col: str | None = None) -> Dataset:
        """
        Method to return a Dataset object from a database selected table
        Inputs:
            - table:
            - schema: Table or dictionary with the table information, column and datatypes
            - schema_name: The name of the database schema where the table exists
            - index_col: A column to index the partitions of the distributed system.
                        If none is provided the primary_key is assumed as the index.
        """
    def get_table_sample(self, table: str, schema: Table | dict | None = None, schema_name: str | None = None, sample_size: int = ...) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by building a query
        with the table name and by adding a suffix to the provided query."""
    def query(self, query: str, schema: Table | dict | None = None) -> Dataset:
        """
        Method that returns a Dataset which results from a provided query
        inputs:
            - query: str - the query syntax
            - schema: the schema with the columns and datatypes for the query output
        """
    def delete_tables(self, tables: list[str]):
        """Delete all rows from tables.

        Args:
            tables (list[str]): list of tables to clear.
        """
    def query_sample(self, query: str, sample_size: int = ...) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by adding a suffix to
        the provided query."""
    def read_database(self, schema_name: str | None = None, index_cols: dict[str, str] | None = None, lazy: bool = True) -> MultiDataset:
        """Method the return a Multi table setting.

        The method returns ll the tables from a selected database/schema
        """
    def write_table(self, data: Dataset, name: str, schema_name: str | None = None, if_exists: str | WriteMode = ...):
        """Method to write a Dataset into a given table from a database.

        inputs:
            - data (ydata.dataset.Dataset): A Dataset with the information to be written to the table
            - name (str): The name of a SQL table
            - schema_name Optional(str): The name of the schema where the table should be created.
               In case schema_name is None and connector.schema is not None, the set schema is used.
            - if_exists : {'fail', 'replace', 'append'}, default WriteMode.FAIL
                How to behave if the table already exists.

                * fail: Raise a InvalidTableException.
                * replace: Drop the table before inserting new values.
                * append: Insert new values to the existing table.
        """
    def write_database(self, data: MultiDataset, table_names: dict | list | None = None, schema_name: str | None = None, if_exists: str | WriteMode = ...):
        """Method to write a Multidataset into a database.

        inputs:
            - data (ydata.dataset.MultiDataset): A MultiDataset with the database information - table_names (
        str): A list or dictionary with the name of the output tables. If a list is provided the order of the
        original tables is assumed based on the original dataset reading order.
            - schema_name Optional(str): The name of the schema where the table should be created
            - if_exists : {'fail', 'replace', 'append'}, default WriteMode.FAIL
                How to behave if the table already exists.

                * fail: Raise a InvalidTableException.
                * replace: Drop the table before inserting new values.
                * append: Insert new values to the existing table.
        """
    def test(self) -> None: ...

class MySQLConnector(RDBMSConnector):
    """MYSQL connector class definition."""
    DB_TYPE: str
    DB_PYTHON_LIBRARY: str

class PostgreSQLConnector(RDBMSConnector):
    """PostGreSQL connector class definition."""
    DB_TYPE: str
    DB_PYTHON_LIBRARY: str

class SnowflakeConnector(RDBMSConnector):
    """Snowflake connector class definition."""
    DB_TYPE: str
    DB_PYTHON_LIBRARY: str
    def list_schemas(self):
        """Method to return the list of available schemas."""
    def set_schema(self, schema_name: str):
        """Method to set the use of a schema from a snowflake database."""
    def get_table(self, table: str, schema_name: str | None = None, schema: Table | dict | list = None) -> Dataset:
        """
            Returns a Dataset with all the records from a selected schema database
            inputs:
            - table (str): The name of the table that we want to retrieve the records
            - schema: A Table or a dict with the table schema (name of the columns and variabletype)
            - schema_name (str): The name of the schema where the table exists
        """
    def write_table(self, data: Dataset, name: str, schema_name: str | None = None, if_exists: str | WriteMode = ...): ...

class AzureSQLConnector(RDBMSConnector):
    """Azure SQL connector class definition."""
    DB_TYPE: str
    DB_PYTHON_LIBRARY: str
    def query_sample(self, query: str, sample_size: int = ...) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by building a query
                with the table name and by adding a suffix to the provided query."""
