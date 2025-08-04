"""Python file that defines the logic for the Multidataset object."""
import os

from copy import deepcopy
from typing import Callable, Iterable, Union

from yaml import safe_load
from pandas import read_csv, read_parquet
from networkx import DiGraph, Graph, connected_components
from pandas import DataFrame

from ydata.dataset import Dataset
from ydata.dataset.filetype import FileType
from ydata.dataset.engines import VALID_ENGINES
from ydata.dataset.schemas import MultiTableSchema, RDBMSSchema, RelationType
from ydata.utils.configuration import TextStyle


class MultiDataset:

    def __init__(
        self,
        datasets: dict[str, Dataset | VALID_ENGINES] | None = None,
        connector: "RDBMSConnector | None" = None,  # noqa: F82
        schema: MultiTableSchema | RDBMSSchema | dict | None = None,
        index_cols: dict[str] | None = None,
        lazy: bool = True
    ) -> None:
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

        self.rdbms_schema = None
        self._datasets = {}
        self._connector = connector

        self.__validate_inputs(datasets=datasets,
                               schema=schema,
                               connector=connector)

        if datasets is not None and schema is not None:
            if type(schema) is dict:  # It is required to use type instead of isinstance,
                # as using isintance it returns True whenever passing a MultiTableSchem
                schema = MultiTableSchema(schema)
            elif isinstance(schema, RDBMSSchema):
                self.rdbms_schema = schema
                schema = MultiTableSchema(schema)

            schema = schema.filter(list(datasets.keys()))

            self._datasets = {
                t: d if isinstance(d, Dataset) else Dataset(
                    d, schema[t].columns)
                for t, d in datasets.items()
            }

        elif connector is not None:
            if schema is None:
                self.rdbms_schema = self._connector.get_database_schema()
                schema = MultiTableSchema(self.rdbms_schema)
            else:
                if isinstance(schema, RDBMSSchema):
                    self.rdbms_schema = schema
                    schema = MultiTableSchema(schema)
                else:
                    # getting the RDBMS schema from the database

                    self.rdbms_schema = self._connector.get_database_schema()

                    diff_tables = list(
                        set(list(self.rdbms_schema.tables)) - set(list(schema.keys())))
                    if len(diff_tables) > 0:
                        for table in diff_tables:
                            self.rdbms_schema.tables.pop(table, None)

            new_tables = [
                t for t in self.rdbms_schema.tables.keys()
                if t not in self._datasets or self._datasets[t] is None
            ]

            if lazy:
                self._datasets.update({t: None for t in new_tables})
            else:
                flt_schema = deepcopy(schema).dict()
                flt_schema = {table: k for table,
                              k in flt_schema.items() if table in new_tables}

                self._datasets.update(self._connector.get_tables(
                    new_tables,
                    schema=flt_schema,
                    schema_name=self.rdbms_schema.name))

        else:
            raise RuntimeError(
                "Either the datasets or a RDBMS connector must be supplied.")

        self._schema = schema

        self._observers_for_new_tables = []

    @staticmethod
    def __validate_inputs(datasets, schema, connector):
        """Auxiliar function to validate the combination of inputs."""
        if datasets is not None and schema is None:
            raise RuntimeError("Schema is a mandatory input whenever providing datasets. "
                               "In order to create a MultiTable dataset the properties and relations between tables must be explicit.")

        if datasets is not None and connector is not None:
            raise RuntimeError(
                "You must provided only one of the following inputs: datasets or connector")

        if datasets is None and connector is None:
            raise RuntimeError(
                "You must provide at least one of the following inputs: datasets or connector.")

    def add_observer_for_new_tables(self, func: Callable):
        """
        Registers an observer function to be notified when new tables are loaded into the MultiDataset.

        Typically used by MultiMetadata to receive updates when deferred tables are materialized.

        Args:
            func (Callable): A callback function that accepts (table_name, Dataset).
        """
        self._observers_for_new_tables.append(func)

    def add_foreign_key(self, table: str, column: str,
                        parent_table: str, parent_column: str,
                        relation_type: str | RelationType = RelationType.MANY_TO_MANY):
        """
        Adds a foreign key relationship to the schema.

        Args:
            table (str): Name of the child table.
            column (str): Foreign key column in the child table.
            parent_table (str): Name of the parent table.
            parent_column (str): Primary key column in the parent table.
            relation_type (str | RelationType): Type of relationship (e.g., MANY_TO_MANY). Defaults to MANY_TO_MANY.
        """
        self.schema.add_foreign_key(
            table, column, parent_table, parent_column, relation_type)

    def add_primary_key(self, table: str, column: str):
        """
        Adds a primary key column to a specific table in the schema.

        Args:
            table (str): Table name.
            column (str): Column name to mark as the primary key.
        """
        self.schema.add_primary_key(table, column)

    @property
    def schema(self):
        """
        Returns the schema associated with the MultiDataset.

        Returns:
            MultiTableSchema: The object defining table structures and relationships.
        """
        return self._schema

    def _get_table_relationships(self):
        fks = []
        for table in self.schema.values():
            for fk in table.foreign_keys:
                fks.append((fk.table, fk.parent_table))
        return fks

    def get_components(self):
        graph = Graph()
        graph.add_nodes_from(self.schema.keys())
        graph.add_edges_from(self._get_table_relationships())

        return connected_components(graph)

    def get_database_dag(self, reverse: bool = False) -> DiGraph:
        graph = DiGraph()
        graph.add_nodes_from(self.schema.keys())
        relationships = self._get_table_relationships()
        if reverse:
            relationships = [(r[1], r[0]) for r in relationships]
        graph.add_edges_from(relationships)
        return graph

    def _fetch_deferred_data(self, tables: list[str]):
        """This method lazily fetches tables and sends the data to the
        MultiMetadata objects registered as observers so that they can compute
        the respective Metadata."""
        for t in tables:
            self._datasets[t] = self._connector.get_table(
                t, schema=self.schema[t].columns, schema_name=self.rdbms_schema.name)
            for observer_fn in self._observers_for_new_tables:
                try:
                    observer_fn(t, self._datasets[t])
                except Exception:
                    self._observers_for_new_tables.remove(observer_fn)

    def _deferred_request_endpoint(self):
        """This method returns an object that will be used by MultiMetadata to
        request a table that was not yet fetched."""
        class DeferredRequest:
            @staticmethod
            def request_table(table: str):
                _ = self[table]
        return DeferredRequest()

    def compute(self):
        """
        Materializes all deferred tables in the dataset by fetching them via the connector, if available.

        Returns:
            MultiDataset: The same object with all tables loaded into memory.
        """
        if self._connector is not None:
            tables_to_fetch = [
                t for t, d in self._datasets.items() if d is None]
            self._fetch_deferred_data(tables_to_fetch)
        return self

    def select_tables(self, tables: Iterable[str]):
        """
        Selects a subset of tables from the MultiDataset.

        Args:
            tables (Iterable[str]): Names of the tables to include.

        Returns:
            MultiDataset: A new MultiDataset instance containing only the selected tables.
        """
        datasets = {k: v for k, v in self._datasets.items() if k in tables}

        if self.rdbms_schema:
            schema = deepcopy(self.rdbms_schema)
            schema.tables = {k: v for k,
                             v in schema.tables.items() if k in tables}
        else:
            schema = deepcopy(self._schema)
            schema = {k: v for k, v in schema.items() if k in tables}

        return MultiDataset(datasets=datasets, schema=schema)

    @classmethod
    def from_files(cls,
                   folder_path: str,
                   schema_path: str,
                   sep=',',
                   file_type=FileType.CSV):
        """
        Load a MultiDataset from a folder of CSV or Parquet files,
        using an optional schema.yaml file to define table relationships.

        Args:
            folder_path (str): Path to the folder containing .csv/.parquet files and a schema.yaml file.

        Returns:
            MultiDataset: An initialized MultiDataset object.
        """
        file_type = FileType(file_type)

        datasets = {}

        for f in os.listdir(folder_path):
            full_path = os.path.join(folder_path, f)

            if file_type == FileType.CSV and f.endswith(".csv"):
                df = read_csv(full_path,
                              sep=sep)
                table_name = f.removesuffix(".csv")
            elif file_type == "parquet" and f.endswith(".parquet"):
                df = read_parquet(full_path)
                table_name = f.removesuffix(".parquet")
            else:
                continue

            datasets[table_name] = Dataset(df)

        # Load schema.yaml
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"{schema_path} is not a valid file path.")

        with open(schema_path, "r") as f:
            try:
                schema = safe_load(f)
            except Exception:
                raise Exception('Please validate the provided schema. It must be a valid yaml file.')

        return MultiDataset(datasets, schema=schema)

    def __getitem__(self, key: str | list[str]) -> Union[Dataset, "MultiDataset"]:
        """
        Access a table or subset of tables from the MultiDataset.

        Args:
            key (str | list[str]): Table name or list of table names.

        Returns:
            Dataset | MultiDataset: A single Dataset if one table is requested, or a MultiDataset if multiple are specified.
        """
        if self._connector is not None:
            all_tables = key if isinstance(key, list) else [key]
            tables_to_fetch = [
                t for t in all_tables if self._datasets[t] is None]
            self._fetch_deferred_data(tables_to_fetch)

        if isinstance(key, list):
            return self.select_tables(key)

        return self._datasets[key]

    def __setitem__(self, key: str, data: Dataset):
        """
        Assigns a new Dataset to a specific table name in the MultiDataset.

        Args:
            key (str): Table name.
            data (Dataset): Dataset to assign.
        """
        self._datasets[key] = data

    def items(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
        return self._datasets.items()

    def keys(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
        return self._datasets.keys()

    def values(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
        return self._datasets.values()

    def __iter__(self):
        """
        Standard dictionary-like methods for iterating over tables in the MultiDataset.
        """
        return self._datasets.__iter__()

    def __str__(self):
        n_tables = len(self.schema)
        str_repr = TextStyle.BOLD + "MultiDataset Summary \n \n" + TextStyle.END

        # get the total number of tables in the DB schema
        str_repr += (
            TextStyle.BOLD
            + "Number of tables: "
            + TextStyle.END
            + f"{n_tables} \n \n"
        )

        # Calculate the summary of the information to be shown
        summary = []

        for table, table_details in self.schema.items():
            pk = table_details.primary_keys
            fk = [key.column for key in table_details.foreign_keys]

            table_summary = {"Table name": table,
                             "Num cols": len(table_details.columns),
                             "Num rows": self[table].nrows if self._datasets[table]
                             is not None else "Number of rows not yet computed",
                             "Primary keys": pk if len(pk) else '',
                             "Foreign keys": fk if len(fk) else '',
                             "Notes": ""}

            summary.append(table_summary)

        str_repr += DataFrame(summary).to_string()

        return str_repr
