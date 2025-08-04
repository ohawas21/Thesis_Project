from _typeshed import Incomplete
from dataclasses import dataclass

logger: Incomplete

@dataclass(frozen=True)
class ForeignKey:
    """Dataclass to define the ForeignKey information."""
    column: str
    parent: str
    def __init__(self, column, parent) -> None: ...

@dataclass(frozen=True)
class TableColumn:
    """Class to store the information of a Column table."""
    name: str
    variable_type: str
    primary_key: bool
    is_foreign_key: bool
    foreign_keys: list
    nullable: bool
    format: str | None = ...
    @staticmethod
    def from_database_column(column, table):
        """Method to return a Table column definition."""
    def __init__(self, name, variable_type, primary_key, is_foreign_key, foreign_keys, nullable, format=...) -> None: ...

@dataclass(frozen=True)
class Table:
    """Class to store the table columns information."""
    name: str
    columns: list[TableColumn]
    primary_keys: list[TableColumn]
    foreign_keys: list[TableColumn]
    @staticmethod
    def from_database_table(table, columns):
        """Method to return a table schema."""
    @property
    def dtypes(self): ...
    def __init__(self, name, columns, primary_keys, foreign_keys) -> None: ...

@dataclass
class Schema:
    """Class to store the database schema information."""
    name: str
    tables: dict
    def __init__(self, name, tables) -> None: ...
