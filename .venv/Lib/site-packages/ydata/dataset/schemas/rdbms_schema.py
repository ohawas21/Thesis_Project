"""Python file with the definition of the different schemas used across YData
Fabric."""
from dataclasses import dataclass
from datetime import time
from os import getenv
from typing import List

from numpy import dtype
from pandas.api.types import is_integer_dtype

from ydata.utils.data_types import VariableType
from ydata.utils.logger import utilslogger_config
from ydata.utils.type_inference import TypeConverter

logger = utilslogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


@dataclass(frozen=True)
class ForeignKey:
    """Dataclass to define the ForeignKey information."""

    column: str
    parent: str


@dataclass(frozen=True)
class TableColumn:
    """Class to store the information of a Column table."""

    name: str
    variable_type: str
    primary_key: bool
    is_foreign_key: bool
    foreign_keys: list
    nullable: bool
    format: str | None = None

    @staticmethod
    def from_database_column(column, table):
        """Method to return a Table column definition."""
        # explore where best can I do this instead

        converter = TypeConverter()

        try:

            python_type = column.type.python_type
            if python_type == time:
                python_type = str

            if python_type is dict: #required for JSON dtypes coming form RDBMS databases
                python_type = str

            if column.nullable and is_integer_dtype(python_type):
                python_type = dtype("float32")

            vartype = converter.from_low(python_type)
        except NotImplementedError:
            logger.info(
                f"[RDBMS SCHEMA]: Python type for {column.name} was not possible to get.")
            vartype = 'string'

        return TableColumn(
            name=column.name,
            variable_type=VariableType(vartype),
            primary_key=column.primary_key,
            is_foreign_key=bool(len(column.foreign_keys) > 0),
            foreign_keys=[
                ForeignKey(f"{table.name}.{column.name}", str(key.column))
                for key in column.foreign_keys
            ],
            nullable=column.nullable,
        )


@dataclass(frozen=True)
class Table:
    """Class to store the table columns information."""

    name: str
    columns: List[TableColumn]
    primary_keys: List[TableColumn]
    foreign_keys: List[TableColumn]

    @staticmethod
    def from_database_table(table, columns):
        """Method to return a table schema."""
        return Table(
            name=table.name,
            columns=columns,
            primary_keys=[
                col for col in columns if col.primary_key
            ],  # Build the foreign_key logic here
            foreign_keys=[col for col in columns if col.is_foreign_key],
        )

    @property
    def dtypes(self):
        return {col.name: col.variable_type for col in self.columns}


@dataclass
class Schema:
    """Class to store the database schema information."""

    name: str
    tables: dict
