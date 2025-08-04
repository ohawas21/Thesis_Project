from _typeshed import Incomplete
from networkx import DiGraph
from typing import Callable, Iterable
from ydata.dataset import Dataset
from ydata.dataset.engines import VALID_ENGINES
from ydata.dataset.schemas import MultiTableSchema, RDBMSSchema, RelationType

class MultiDataset:
    rdbms_schema: Incomplete
    def __init__(self, datasets: dict[str, Dataset | VALID_ENGINES] | None = None, connector: RDBMSConnector | None = None, schema: MultiTableSchema | RDBMSSchema | dict | None = None, index_cols: dict[str] | None = None, lazy: bool = True) -> None:
        """
        Initialize a MultiDataset object from either:
        - A dictionary of table names to Datasets or compatible dataframes, along with a schema.
        - An RDBMS connector with an optional schema.

        Args:
            datasets (dict[str, Dataset | Dask.DataFrame | pandas.DataFrame] | None): A dictionary of table names to pandas, Dask, or Dataset objects.
            connector (RDBMSConnector | None): Optional database connector used to lazily fetch data from a relational database.
            schema (MultiTableSchema | RDBMSSchema | dict | None): Schema defining the structure and relationships of the dataset.
            index_cols (dict[str] | None): Optional index column definitions for the tables.
            lazy (bool): Whether to defer loading data from the connector until accessed. Defaults to True.

        Raises:
            RuntimeError: If neither datasets nor a connector is provided, or if schema is missing when datasets are given.
        """
    def add_observer_for_new_tables(self, func: Callable):
        """
        Registers an observer function to be notified when new tables are loaded into the MultiDataset.

        Typically used by MultiMetadata to receive updates when deferred tables are materialized.

        Args:
            func (Callable): A callback function that accepts (table_name, Dataset).
        """
    def add_foreign_key(self, table: str, column: str, parent_table: str, parent_column: str, relation_type: str | RelationType = ...):
        """
        Adds a foreign key relationship to the schema.

        Args:
            table (str): Name of the child table.
            column (str): Foreign key column in the child table.
            parent_table (str): Name of the parent table.
            parent_column (str): Primary key column in the parent table.
            relation_type (str | RelationType): Type of relationship (e.g., MANY_TO_MANY). Defaults to MANY_TO_MANY.
        """
    def add_primary_key(self, table: str, column: str):
        """
        Adds a primary key column to a specific table in the schema.

        Args:
            table (str): Table name.
            column (str): Column name to mark as the primary key.
        """
    @property
    def schema(self):
        """
        Returns the schema associated with the MultiDataset.

        Returns:
            MultiTableSchema: The object defining table structures and relationships.
        """
    def get_components(self): ...
    def get_database_dag(self, reverse: bool = False) -> DiGraph: ...
    def compute(self):
        """
        Materializes all deferred tables in the dataset by fetching them via the connector, if available.

        Returns:
            MultiDataset: The same object with all tables loaded into memory.
        """
    def select_tables(self, tables: Iterable[str]):
        """
        Selects a subset of tables from the MultiDataset.

        Args:
            tables (Iterable[str]): Names of the tables to include.

        Returns:
            MultiDataset: A new MultiDataset instance containing only the selected tables.
        """
    @classmethod
    def from_files(cls, folder_path: str, schema_path: str, sep: str = ',', file_type=...):
        """
        Load a MultiDataset from a folder of CSV or Parquet files,
        using an optional schema.yaml file to define table relationships.

        Args:
            folder_path (str): Path to the folder containing .csv/.parquet files and a schema.yaml file.

        Returns:
            MultiDataset: An initialized MultiDataset object.
        """
    def __getitem__(self, key: str | list[str]) -> Dataset | MultiDataset:
        """
        Access a table or subset of tables from the MultiDataset.

        Args:
            key (str | list[str]): Table name or list of table names.

        Returns:
            Dataset | MultiDataset: A single Dataset if one table is requested, or a MultiDataset if multiple are specified.
        """
    def __setitem__(self, key: str, data: Dataset):
        """
        Assigns a new Dataset to a specific table name in the MultiDataset.

        Args:
            key (str): Table name.
            data (Dataset): Dataset to assign.
        """
    def items(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
    def keys(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
    def values(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
    def __iter__(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
