"""File to set the class for all the RDBMS like connectors."""

# Improve the SQL connectors for better outputs

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Union

from dask import dataframe as dddataframe
from pandas import read_sql_query
from snowflake.sqlalchemy import URL
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.exc import ArgumentError
from tqdm.auto import tqdm

from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.exceptions import (InvalidDatabaseConnection, InvalidIndexCol, InvalidQuery, InvalidTable,
                                         InvalidTableException)
from ydata.connectors.storages import _MAX_SAMPLE, _RDBMS_STORAGE
from ydata.core.connectors import WriteMode
from ydata.dataset import Dataset
from ydata.dataset.dataset import Schema as DatasetSchema
from ydata.dataset.multidataset import MultiDataset
from ydata.dataset.schemas import RDBMSSchema as Schema
from ydata.dataset.schemas import Table
from ydata.dataset.schemas.rdbms_schema import TableColumn
from ydata.utils.data_types import VariableType

import pandas as pd

@dataclass
class ConnectionString:
    """Connection string settings."""
    username: str
    password: str
    database: str
    schema: Optional[str] = None
    port: Optional[int] = None
    hostname: Optional[str] = None
    warehouse: Optional[str] = None


# @typechecked
class RDBMSConnector(BaseConnector):
    """RDBMS connectors structure."""

    STORAGE_TYPE = _RDBMS_STORAGE
    DB_TYPE = None
    DB_PYTHON_LIBRARY = None
    _SAMPLE_SUFFIX = " ORDER BY RAND() LIMIT {};"

    _conn = None
    _echo: bool = True
    _uri = None
    _schema = None
    _schema_name = None

    def __init__(self, conn_string: Union[ConnectionString, dict], echo: bool = False) -> None:
        super().__init__()
        self.conn_str = (
            conn_string
            if isinstance(conn_string, ConnectionString)
            else ConnectionString(**conn_string)
        )
        self._database = self.conn_str.database

        self._echo = echo

    @property
    def uri(self):
        if self._uri is None:
            self._uri = self._build_uri()

        return self._uri

    @property
    def database(self):
        """Returns the name of the selected database to connect to."""
        return self._database

    @property
    def connection(self):
        """Returns the connection engine."""
        if self._conn is None:
            self._conn = create_engine(self.uri, echo=self._echo)

        return self._conn

    def _build_uri(self):
        if self.DB_TYPE == 'mssql':
            if self.conn_str.port is None:
                uri = (f"{self.DB_TYPE}+{self.DB_PYTHON_LIBRARY}://{self.conn_str.username}:{self.conn_str.password}@{self.conn_str.hostname},{self.conn_str.port}/{self._database}?"
                       f"driver=ODBC+Driver+17+for+SQL+Server")
            else:
                uri =  (f"{self.DB_TYPE}+{self.DB_PYTHON_LIBRARY}://{self.conn_str.username}:{self.conn_str.password}@{self.conn_str.hostname}/{self._database}?"
                        f"driver=ODBC+Driver+17+for+SQL+Server")
        else:
            if self.conn_str.port is not None:
                uri = f"{self.DB_TYPE}+{self.DB_PYTHON_LIBRARY}://{self.conn_str.username}:{self.conn_str.password}@" \
                      f"{self.conn_str.hostname}:{self.conn_str.port}/{self._database}"
            else:
                uri = f"{self.DB_TYPE}+{self.DB_PYTHON_LIBRARY}://{self.conn_str.username}:{self.conn_str.password}@" \
                      f"{self.conn_str.hostname}/{self._database}"
        return uri

    def _get_query_schema(self, query):
        """Returns the dataframe schema for a given provided query."""
        try:
            iter_data = read_sql_query(
                sql=query, con=self.connection, chunksize=50000)
            schema = dict(next(iter_data).dtypes)
        except ArgumentError as exc:
            raise InvalidDatabaseConnection(
                "Invalid database connection.") from exc
        except Exception as exc:
            raise InvalidQuery(f"Invalid query provided. {exc}") from exc
        # todo convert schema from pandas schema to YData expected VariableTypes
        return schema

    def set_schema(self, schema_name: str | None = None):
        """
        Method to set the schema name to be used by the connector
        Parameters
        ----------
        schema_name A string with the name of the database schema
        """
        self._schema_name = schema_name

    @property
    def schema_name(self):
        return self._schema_name

    def get_database_schema(self, schema_name: Optional[str] = None, refresh: bool = True, set_schema: bool = True) -> Schema:
        """Method to calculate the schema of the selected database to be
        queried."""
        if self._schema is not None and not refresh:
            return self._schema

        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        # Todo convert to YData expected variables types
        self._metadata = MetaData(schema=schema_name)
        self._metadata.reflect(self.connection)

        tables = {}
        for table in self._metadata.sorted_tables:
            cols = []
            for col in table.columns.values():
                cols.append(TableColumn.from_database_column(col, table))
            table = Table.from_database_table(table, cols)
            tables[table.name] = table

        schema = Schema(
            name=self._database if schema_name is None else schema_name, tables=tables)
        if set_schema:
            self._schema = schema
        return schema

    def dispose(self) -> None:
        """Dispose opened connections."""
        self.connection.dispose()

    def get_tables(self, tables: list[str],
                   schema: Schema | dict | None = None,
                   schema_name: Optional[str] = None):
        """Method to return a v   object from a database This method can return
        the whole database in case tables parameter is not provided."""

        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        if isinstance(schema, Schema):
            dbschema = schema
            tables_dtypes = {table: v.dtypes for table,
                             v in dbschema.tables.items()}
        elif isinstance(schema, dict):
            tables_dtypes = {table: v['columns']
                             for table, v in schema.items()}
        else:
            dbschema = self.get_database_schema(schema_name=schema_name)
            tables_dtypes = {table: v.dtypes for table,
                             v in dbschema.tables.items()}

        datasets = {}
        for table in tables:
            dtypes = tables_dtypes[table]
            datasets[table] = self.get_table(
                table, schema=dtypes, schema_name=schema_name)

        return datasets

    def _build_query_table(self, table_name: str, schema_name: str | None = None) -> str:
        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        if schema_name:
            query = f"SELECT * FROM {schema_name}.{table_name}"
        else:
            query = f"SELECT * FROM {table_name}"
        return query

    def get_table(
        self,
        table: str,
        schema: Table | dict | None = None,
        schema_name: str | None = None,
        index_col: str | None = None,
    ) -> Dataset:
        """
        Method to return a Dataset object from a database selected table
        Inputs:
            - table:
            - schema: Table or dictionary with the table information, column and datatypes
            - schema_name: The name of the database schema where the table exists
            - index_col: A column to index the partitions of the distributed system.
                        If none is provided the primary_key is assumed as the index.
        """
        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        return self._get_table(table=table,
                               schema=schema,
                               schema_name=schema_name)

    def _get_table(
        self,
        table: str,
        schema: Table | dict = None,
        schema_name: str | None = None,
    ) -> Dataset:
        try:
            if isinstance(schema, Schema) is False:
                if schema_name:
                    dbschema = self.get_database_schema(
                        schema_name=schema_name, refresh=False)
                else:
                    dbschema = self.get_database_schema(refresh=False)
                    schema_name = dbschema.name
            else:
                dbschema = schema

            table = dbschema.tables[table]
        except KeyError as exc:
            raise InvalidTable(
                f"Invalid table provided {table}. "
                f"Please provide a valid column from the schema."
            ) from exc
        except ArgumentError as exc:
            raise InvalidDatabaseConnection(
                "Invalid database connection.") from exc

        query = self._build_query_table(table_name=table.name,
                                        schema_name=schema_name)

        return self._query(query, schema=schema)

    def get_table_sample(
        self, table: str, schema: Table | dict | None = None, schema_name: str | None = None, sample_size: int = _MAX_SAMPLE
    ) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by building a query
        with the table name and by adding a suffix to the provided query."""

        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        query = self._build_query_table(
            schema_name=schema_name, table_name=table) + self._SAMPLE_SUFFIX.format(sample_size)

        return self._query(query, schema=schema)

    def query(self, query: str, schema: Union[Table, dict] | None = None) -> Dataset:
        """
        Method that returns a Dataset which results from a provided query
        inputs:
            - query: str - the query syntax
            - schema: the schema with the columns and datatypes for the query output
        """
        return self._query(query, schema=schema)

    def _query(self, query: str, schema: Union[Table, dict] | None = None) -> Dataset:
        if isinstance(schema, Table):
            schema = {col.name: col.variable_type for col in schema.columns}

        try:
            con = self.connection.raw_connection().dbapi_connection
            iter_data = read_sql_query(
                sql=query, con=con, chunksize=50000)

            data = dddataframe.multi.concat(
                [dddataframe.from_pandas(chunk, npartitions=3)
                 for chunk in iter_data]
            )

        except Exception as exc:
            raise InvalidQuery(f"Invalid query provided. {exc}") from exc

        return Dataset(data, schema=schema)

    def _build_delete_query(self, table_name: str, schema_name: str | None = None) -> str:
        """Creates a query to delete all rows from table.

        Args:
            table_name (str): table to be deleted
            schema_name (str | None, optional): schema that contains the table. Defaults to None.

        Returns:
            str: delete query
        """
        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        if schema_name:
            query = f"DELETE FROM {self.database}.{schema_name}.{table_name}"
        else:
            query = f"DELETE FROM {self.database}.{table_name}"
        return query

    def delete_tables(self, tables: list[str]):
        """Delete all rows from tables.

        Args:
            tables (list[str]): list of tables to clear.
        """
        schema = self.get_database_schema()
        for table in tables:
            # check if table exists prior to deleting the records
            if table in schema.tables:
                with self.connection.connect() as connection:
                    connection.execute(text(self._build_delete_query(table)))

    def query_sample(
        self,
        query: str,
        sample_size: int = _MAX_SAMPLE,
    ) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by adding a suffix to
        the provided query."""
        sample_suffix = self._SAMPLE_SUFFIX.format(sample_size)

        if query.endswith(";"):
            query = query.replace(";", "", 1) + sample_suffix
        else:
            query += sample_suffix
        return self.query(query=query)

    def read_database(
        self,
        schema_name: Optional[str] = None,
        index_cols: Optional[Dict[str, str]] = None,
        lazy: bool = True
    ) -> MultiDataset:
        """Method the return a Multi table setting.

        The method returns ll the tables from a selected database/schema
        """
        if index_cols is None:
            index_cols = {}

        if schema_name:
            self.set_schema(schema_name)

        return MultiDataset(connector=self, lazy=lazy, index_cols=index_cols)

    def write_table(self, data: Dataset,
                    name: str,
                    schema_name: Optional[str] = None,
                    if_exists: str | WriteMode = WriteMode.FAIL):
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
        # Convert the dataset into DASK
        dd_data = data.to_dask()
        if_exists = WriteMode(if_exists).value

        # Checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        try:
            with self.connection.connect() as connection:
                for i in range(dd_data.npartitions):
                    partition = dd_data.get_partition(i).compute()
                    if i == 0:
                        partition.to_sql(name=name, con=connection, schema=schema_name, if_exists=if_exists, index=False)
                    if i > 0:
                        partition.to_sql(name=name, con=connection, schema=schema_name, if_exists='append', index=False)

        except ValueError as exc:
            raise InvalidTableException(
                f"The provided table name {name} already exists. "
                f"Please validate the selected schema or provide a different name."
            ) from exc

    def write_database(self, data: MultiDataset,
                       table_names: dict | list | None = None,
                       schema_name: str | None = None,
                       if_exists: str | WriteMode = WriteMode.FAIL):
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
        if_exists = WriteMode(if_exists).value

        # checks whether user as set a schema otherwise uses a default in case it exists
        schema_name = schema_name if schema_name is not None else self.schema_name

        # compute table_name if not provided and schema_name present
        if table_names is None:
            table_names = list(data.schema.keys())

        if isinstance(table_names, list):
            table_names = dict(zip(data, table_names))

        for (t_name, out_t_name) in tqdm(table_names.items(), desc="Writing tables in the database"):
            try:
                table_data = data[t_name]

                self.write_table(table_data, out_t_name,
                                 schema_name=schema_name, if_exists=if_exists)
            except KeyError:
                warnings.warn(f"The provided table {t_name} does not exist in the original database. "
                              f"This table will be skipped.")

    def test(self):
        self.connection.connect()


class MySQLConnector(RDBMSConnector):
    """MYSQL connector class definition."""

    DB_TYPE = "mysql"
    DB_PYTHON_LIBRARY = "pymysql"


class PostgreSQLConnector(RDBMSConnector):
    """PostGreSQL connector class definition."""

    DB_TYPE = "postgresql"
    DB_PYTHON_LIBRARY = "psycopg2"
    _SAMPLE_SUFFIX = " ORDER BY RANDOM() LIMIT {};"
    _schema_name = "public"  # This is PostGreSQL default schema


class SnowflakeConnector(RDBMSConnector):
    """Snowflake connector class definition."""

    DB_TYPE = "snowflake"
    DB_PYTHON_LIBRARY = "snowflake"
    _SAMPLE_SUFFIX = " SAMPLE BERNOULLI ({} rows)"
    _schema_name = 'PUBLIC'  # this is snowflake's default schema

    _REGEX = re.compile(
        r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)', re.IGNORECASE)

    def _build_uri(self):
        db_parameters = {
            'user': self.conn_str.username,
            'password': self.conn_str.password,
            'warehouse': self.conn_str.warehouse,
            'database': self.conn_str.database,
            'port': self.conn_str.port,
            'application': 'ydata_fabric'
        }

        if re.match(SnowflakeConnector._REGEX, self.conn_str.hostname) is not None:
            db_parameters['account'] = self.conn_str.hostname.split('.')[0]
            db_parameters['host'] = self.conn_str.hostname
        else:  # assume that host is the account identifier
            db_parameters['account'] = self.conn_str.hostname

        return URL(**db_parameters)

    def list_schemas(self):
        """Method to return the list of available schemas."""
        with self.connection.connect() as connection:
            return list(connection.execute(text("SHOW SCHEMAS")))

    def set_schema(self, schema_name: str):
        """Method to set the use of a schema from a snowflake database."""
        self._schema_name = schema_name

        with self.connection.connect() as connection:
            connection.execute(text(f"USE SCHEMA {schema_name}"))

    def get_table(
        self, table: str,
        schema_name: Optional[str] = None,
        schema: Union[Table, dict, list] = None,
    ) -> Dataset:
        """
            Returns a Dataset with all the records from a selected schema database
            inputs:
            - table (str): The name of the table that we want to retrieve the records
            - schema: A Table or a dict with the table schema (name of the columns and variabletype)
            - schema_name (str): The name of the schema where the table exists
        """
        if table.isupper():
            table = table.lower()
            # Snowflake's metadata is returned in upper cases if the table names follow these
            # upper-case best practices. Otherwise, is case-sensitive.

        return self._get_table(
            table=table,
            schema_name=schema_name,
        )

    def write_table(self, data: Dataset,
                    name: str,
                    schema_name: Optional[str] = None,
                    if_exists: str | WriteMode = WriteMode.FAIL):

        name = name.upper()  # this is required to comply with Snowflake's partners best practices
        super().write_table(data=data, name=name,
                            schema_name=schema_name, if_exists=if_exists)


class AzureSQLConnector(RDBMSConnector):
    """Azure SQL connector class definition."""

    DB_TYPE = "mssql"
    DB_PYTHON_LIBRARY = "pyodbc"
    _SAMPLE_SUFFIX = " TABLESAMPLE ({} ROWS);"
    _schema_name = "dbo"  # This is Azure's default schema

    def query_sample(
        self,
        query: str,
        sample_size: int = _MAX_SAMPLE,
    ) -> Dataset:
        """Returns a sample of data from an RDBMS dataset by building a query
                with the table name and by adding a suffix to the provided query."""
        init_sample_size=sample_size*3

        #build sample query for AzureSQL
        # AZURE DOES NOT HAVE A RANDOM command as PostGreSQL or MySQL and TABLESAMPLE might fail depending on the
        # query complexity
        # The following query ensures a certain level of randomization while still performant for larger tables
        query_sample = f"""
            WITH RandomSubset AS (
                SELECT TOP {init_sample_size} *
                FROM ({query}) AS samplequery
                order by NEWID()
            )
            SELECT top {sample_size} *
            from RandomSubset
            order by NEWID();
        """

        return self._query(query_sample)


def _convert_table_schema_to_dataset_schema(table: Table | None) -> dict[str, DatasetSchema] | None:
    if not table:
        return None

    return {
        col.name: DatasetSchema(column=col.name, vartype=VariableType(
            col.variable_type), format=col.format)
        for col in table.columns
    }
